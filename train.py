import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.autograd as autograd
import numpy as np
import copy
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter
from lib.utils.confuse_matrix import *
from lib.utils.Fourier_Aug import *
from train.utils import *
from train.loss import *
from train.context import disable_tracking_bn_stats
from train.ramps import exp_rampup

def train_fedgm(train_dloader_list, test_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, args, pre_models=None, pre_classifiers=None):
    task_criterion = nn.CrossEntropyLoss().cuda()
    criterion_im = Lim(1e-5)
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    
    # previous global model
    pre_global_model = copy.deepcopy(model_list[1])
    pre_global_classifier = copy.deepcopy(classifier_list[1])

    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    # train local source domain models
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        # Train model locally on source domains
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                    model_list[1:],
                                                                                    classifier_list[1:],
                                                                                    optimizer_list[1:],
                                                                                    classifier_optimizer_list[1:]):

            # check if the source domain is the malicious domain with poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            for i, (image_ws, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break
                image_s = image_ws[0].cuda()
                image_s_s = image_ws[1].cuda()
                label_s = label_s.long().cuda()
                
                # each source domain do optimize
                feature = model(image_s)
                output = classifier(feature)

                feature_aug = model(image_s_s)
                output_aug = classifier(feature_aug)

                src_loss1 = task_criterion(output_aug, label_s)
                src_loss2 = task_criterion(output, label_s)

                # task loss
                task_loss_s = 0.5 * src_loss1 + 0.5 * src_loss2

                # intra-domain gradient matching
                grad_cossim11 = []
                #netE+C1
                for n, p in classifier.named_parameters():
                # for n, p in model.named_parameters():
                    # if len(p.shape) == 1: continue

                    real_grad = grad([src_loss1],
                                        [p],
                                        create_graph=True,
                                        only_inputs=True,
                                        allow_unused=False)[0]
                    fake_grad = grad([src_loss2],
                                        [p],
                                        create_graph=True,
                                        only_inputs=True,
                                        allow_unused=False)[0]
                    
                    if len(p.shape) > 1:
                        _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
                    else:
                        _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)

                    grad_cossim11.append(_cossim)

                grad_cossim1 = torch.stack(grad_cossim11)
                gm_intra_loss = (1.0 - grad_cossim1).mean()

                # # inter-domain gradient matching
                for i in range(1, len(pre_classifiers)):
                    grad_cossim_inter = []
                    feature = model(image_s)
                    output_inter = pre_classifiers[i](feature)
                    inter_loss = task_criterion(output_inter, label_s)
                    
                    for g_p, p in zip(pre_classifiers[i].named_parameters(), classifier.named_parameters()):
                        g_p = g_p[1]
                        p = p[1]
                        inter_grad = grad([inter_loss],
                                            [g_p],
                                            create_graph=True,
                                            only_inputs=True,
                                            allow_unused=False)[0]
                        fake_grad = grad([src_loss1],
                                            [p],
                                            create_graph=True,
                                            only_inputs=True,
                                            allow_unused=False)[0]
                        if len(p.shape) > 1:
                            _cossim_inter = F.cosine_similarity(inter_grad, fake_grad, dim=1).mean()
                        else:
                            _cossim_inter = F.cosine_similarity(inter_grad, fake_grad, dim=0)
                        
                        grad_cossim_inter.append(_cossim_inter)

                    grad_cossim_inter = torch.stack(grad_cossim_inter)
                    if i == 1:
                        gm_inter_loss = (1.0 - grad_cossim_inter).mean()
                    else:
                        gm_inter_loss += (1.0 - grad_cossim_inter).mean()
                
                gm_inter_loss /= (len(pre_classifiers)-1)

                # overall losses: task_loss + gm_inter + gm_intra
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                loss = task_loss_s + args.inter * gm_inter_loss + args.intra * gm_intra_loss
                loss.backward()
                optimizer.step()
                classifier_optimizer.step()

    # save the local source domain models, before model aggregating
    pre_models = []
    pre_classifiers = []
    for i in range(0, len(model_list)):
        pre_models.append(copy.deepcopy(model_list[i]))
        pre_classifiers.append(copy.deepcopy(classifier_list[i]))

    # the weights of local source domain models for aggregating
    domain_weight = []
    num_domains = len(model_list[1:])
    for i in range(num_domains):
        domain_weight.append(1.0/num_domains)
    
    # aggregating the local source domain models by FedAvg
    federated_avg(model_list[1:], domain_weight, mode='fedavg')
    federated_avg(classifier_list[1:], domain_weight, mode='fedavg')

    return pre_models, pre_classifiers, domain_weight

