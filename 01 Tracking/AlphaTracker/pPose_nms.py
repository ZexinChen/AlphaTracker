# -*- coding: utf-8 -*-
import torch
import json
import os
import zipfile
import time
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from opt import opt

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
gamma_avg = 22.48/17
scoreThreds = 0.3
# scoreThreds = 0
matchThreds = 5
matchThreds_ratio = 0.6
areaThres = 0#40 * 40.5
alpha = 0.1
#pool = ThreadPool(4)


def pose_nms(bboxes, bbox_scores, pose_preds, pose_scores):
    '''
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n,)
    pose_preds:     pose locations list (n, 17, 2)
    pose_scores:    pose scores list    (n, 17, 1)
    '''
    #global ori_pose_preds, ori_pose_scores, ref_dists

    pose_scores[pose_scores == 0] = 1e-5

    final_result = []

    ori_bbox = bboxes.clone()
    ori_bbox_scores = bbox_scores.clone()
    ori_pose_preds = pose_preds.clone()
    ori_pose_scores = pose_scores.clone()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(dim=1)
    npose = pose_preds.shape[1]

    human_ids = np.arange(nsamples)
    # print('npose_nms.py:',bboxes.shape,human_ids,nsamples)
    # Do pPose-NMS
    pick = []
    merge_ids = []
    # while(human_scores.shape[0] != 0):
    while(human_ids.shape[0] != 0):
        # Pick the one with highest score
        pick_id = torch.argmax(human_scores)
        # print('npose_nms.py:',bboxes.shape,nsamples,human_ids,pick_id,human_scores.shape)

        pick.append(human_ids[pick_id])
        # num_visPart = torch.sum(pose_scores[pick_id] > 0.2)

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[pick_id]]
        simi = get_parametric_distance(pick_id, pose_preds, pose_scores, ref_dist)
        num_match_keypoints = PCK_match(pose_preds[pick_id], pose_preds, ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        # delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[(simi > gamma) | (num_match_keypoints >= matchThreds)]
        # delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[(simi > gamma) | (num_match_keypoints >= int(npose*matchThreds_ratio))]
        delete_ids = torch.from_numpy(np.arange(human_scores.shape[0]))[((simi)/npose > gamma) | (num_match_keypoints >= int(npose*matchThreds_ratio))]

        # print('pPose-NMS.py:delete_ids:',delete_ids)
        # print('pPose-NMS.py:pick_id:',pick_id)
        # print('pPose-NMS.py:merge_ids:',merge_ids)
        if delete_ids.shape[0] == 0:
            delete_ids = pick_id
        #else:
        #    delete_ids = torch.from_numpy(delete_ids)

        merge_ids.append(human_ids[delete_ids])
        pose_preds = np.delete(pose_preds, delete_ids, axis=0)
        pose_scores = np.delete(pose_scores, delete_ids, axis=0)
        human_ids = np.delete(human_ids, delete_ids)
        human_scores = np.delete(human_scores, delete_ids, axis=0)
        bbox_scores = np.delete(bbox_scores, delete_ids, axis=0)

    assert len(merge_ids) == len(pick)
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    bbox_pick = ori_bbox[pick]
    #final_result = pool.map(filter_result, zip(scores_pick, merge_ids, preds_pick, pick, bbox_scores_pick))
    #final_result = [item for item in final_result if item is not None]
    # print('pPose-NMS.py:pick:',pick)

    for j in range(len(pick)):
        ids = np.arange(4)
        max_score = torch.max(scores_pick[j, ids, 0])

        # print('pPose-NMS.py:max_score1:',max_score)
        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j], ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick[j]])

        max_score = torch.max(merge_score[ids])
        # print('pPose-NMS.py:max_score2:',max_score)
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])
        # print('pPose-NMS.py:merge_pose:',merge_pose)
        # print('pPose-NMS.py:(xmax - xmin) * (ymax - ymin) < areaThres:',(xmax - xmin) * (ymax - ymin) < areaThres)

        if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < areaThres):
            continue

        final_result.append({
            'keypoints': merge_pose - 0.3,
            'kp_score': merge_score,
            'proposal_score': torch.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score),
            'box': bbox_pick[j]
        })

    return final_result


def filter_result(args):
    score_pick, merge_id, pred_pick, pick, bbox_score_pick = args
    global ori_pose_preds, ori_pose_scores, ref_dists
    ids = np.arange(17)
    max_score = torch.max(score_pick[ids, 0])

    if max_score < scoreThreds:
        return None

    # Merge poses
    merge_pose, merge_score = p_merge_fast(
        pred_pick, ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick])

    max_score = torch.max(merge_score[ids])
    if max_score < scoreThreds:
        return None

    xmax = max(merge_pose[:, 0])
    xmin = min(merge_pose[:, 0])
    ymax = max(merge_pose[:, 1])
    ymin = min(merge_pose[:, 1])

    if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < 40 * 40.5):
        return None

    return {
        'keypoints': merge_pose - 0.3,
        'kp_score': merge_score,
        'proposal_score': torch.mean(merge_score) + bbox_score_pick + 1.25 * max(merge_score)
    }


def p_merge(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    '''
    dist = torch.sqrt(torch.sum(
        torch.pow(ref_pose[np.newaxis, :] - cluster_preds, 2),
        dim=2
    ))  # [n, 17]

    kp_num = 17
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = torch.zeros(kp_num, 2)
    final_score = torch.zeros(kp_num)

    if cluster_preds.dim() == 2:
        cluster_preds.unsqueeze_(0)
        cluster_scores.unsqueeze_(0)
    if mask.dim() == 1:
        mask.unsqueeze_(0)

    for i in range(kp_num):
        cluster_joint_scores = cluster_scores[:, i][mask[:, i]]  # [k, 1]
        cluster_joint_location = cluster_preds[:, i, :][mask[:, i].unsqueeze(
            -1).repeat(1, 2)].view((torch.sum(mask[:, i]), -1))

        # Get an normalized score
        normed_scores = cluster_joint_scores / torch.sum(cluster_joint_scores)

        # Merge poses by a weighted sum
        final_pose[i, 0] = torch.dot(cluster_joint_location[:, 0], normed_scores.squeeze(-1))
        final_pose[i, 1] = torch.dot(cluster_joint_location[:, 1], normed_scores.squeeze(-1))

        final_score[i] = torch.dot(cluster_joint_scores.transpose(0, 1).squeeze(0), normed_scores.squeeze(-1))

    return final_pose, final_score


def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    '''
    dist = torch.sqrt(torch.sum(
        torch.pow(ref_pose[np.newaxis, :] - cluster_preds, 2),
        dim=2
    ))

    kp_num = 4
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = torch.zeros(kp_num, 2)
    final_score = torch.zeros(kp_num)

    if cluster_preds.dim() == 2:
        cluster_preds.unsqueeze_(0)
        cluster_scores.unsqueeze_(0)
    if mask.dim() == 1:
        mask.unsqueeze_(0)

    # Weighted Merge
    masked_scores = cluster_scores.mul(mask.float().unsqueeze(-1))
    normed_scores = masked_scores / torch.sum(masked_scores, dim=0)

    final_pose = torch.mul(cluster_preds, normed_scores.repeat(1, 1, 2)).sum(dim=0)
    final_score = torch.mul(masked_scores, normed_scores).sum(dim=0)
    return final_pose, final_score


def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_preds[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    mask = (dist <= 1)

    # Define a keypoints distance
    keypoint_scores.squeeze_()
    if keypoint_scores.dim() == 1:
        keypoint_scores.unsqueeze_(0)
    if pred_scores.dim() == 1:
        pred_scores.unsqueeze_(1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = pred_scores.repeat(1, all_preds.shape[0]).transpose(0, 1)

    score_dists = torch.zeros(all_preds.shape[0], pred_scores.shape[1])
    score_dists[mask] = torch.tanh(pred_scores[mask] / delta1) * torch.tanh(keypoint_scores[mask] / delta1)

    point_dist = torch.exp((-1) * dist / delta2)
    final_dist = torch.sum(score_dists, dim=1) + mu * torch.sum(point_dist, dim=1)

    return final_dist


def PCK_match(pick_pred, all_preds, ref_dist):
    dist = torch.sqrt(torch.sum(
        torch.pow(pick_pred[np.newaxis, :] - all_preds, 2),
        dim=2
    ))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = torch.sum(
        dist / ref_dist <= 1,
        dim=1
    )

    return num_match_keypoints


def write_json(all_results, outputpath, for_eval=True):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    form = opt.format
    json_results = []
    json_results_cmu = {}
    count = 0
    for im_res in all_results:
        im_name = im_res['imgname']
        # for human in im_res['result']:
        for human_id in range(len(im_res['result'])):
            human = im_res['result'][human_id]
            keypoints = []
            result = {}
            if for_eval:
                result['image_id'] = int(im_name.split('/')[-1].split('.')[0].split('_')[-1])
            else:
                result['image_id'] = im_name.split('/')[-1]
            result['file_name'] = im_name.split('/')[-1]
            result['category_id'] = 1
            result['boxes'] = im_res['boxes'].cpu().numpy().tolist()

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)
            result['box'] = human['box'].cpu().numpy().tolist()

            if form == 'cmu': # the form of CMU-Pose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['bodies']=[]
                tmp={'joints':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['joints'].append(result['keypoints'][i])
                    tmp['joints'].append(result['keypoints'][i+1])
                    tmp['joints'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['bodies'].append(tmp)
            elif form == 'open': # the form of OpenPose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['people']=[]
                tmp={'pose_keypoints_2d':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+1])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['people'].append(tmp)
            else:
                json_results.append(result)
                count+=1
                # print('pPose-NMS.py:count:',count)
                # print('pPose-NMS.py:result:',result)

    if form == 'cmu': # the form of CMU-Pose
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    elif form == 'open': # the form of OpenPose
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    else:
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results))


def write_json_withID(all_results, outputpath, img_json, for_eval=True):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    with open(img_json, 'r') as File:
        img_json_data = json.load(File)
    imgname_id_dict = {}
    for img_item in img_json_data['images']:
        imgname_id_dict[img_item['file_name']] = img_item['id']

    form = opt.format
    json_results = []
    json_results_cmu = {}
    count = 0
    for im_res in all_results:
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            result = {}
            # if for_eval:
            #     result['image_id'] = int(im_name.split('/')[-1].split('.')[0].split('_')[-1])
            # else:
            #     result['image_id'] = im_name.split('/')[-1]
            result['image_id'] = imgname_id_dict[im_name.split('/')[-1]]
            result['file_name'] = im_name.split('/')[-1]
            result['category_id'] = 1
            result['boxes'] = im_res['boxes'].cpu().numpy().tolist()

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)

            if form == 'cmu': # the form of CMU-Pose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['bodies']=[]
                tmp={'joints':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['joints'].append(result['keypoints'][i])
                    tmp['joints'].append(result['keypoints'][i+1])
                    tmp['joints'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['bodies'].append(tmp)
            elif form == 'open': # the form of OpenPose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['people']=[]
                tmp={'pose_keypoints_2d':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+1])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['people'].append(tmp)
            else:
                json_results.append(result)
                count+=1
                # print('pPose-NMS.py:count:',count)
                # print('pPose-NMS.py:result:',result)

    if form == 'cmu': # the form of CMU-Pose
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    elif form == 'open': # the form of OpenPose
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    else:
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results))













