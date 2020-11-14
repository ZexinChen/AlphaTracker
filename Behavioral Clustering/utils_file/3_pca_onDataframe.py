
import setting
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_for_faces(show_fig=False):
    arg = setting.args_class()

    print('reading...')
    with open('./faces.pckl','rb') as f:
        faces = pickle.load(f)

    #n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
    print('calculating...')
    faces_pca = PCA(n_components=0.9)
    faces_pca.fit(faces)

    if show_fig:
        fig, axes = plt.subplots(2,5,figsize=(9,3),
            subplot_kw = {'xticks':[], 'yticks':[]},
            gridspec_kw = dict(hspace=0.01, wspace=0.01))

        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_pca.components_[i].reshape(200,200),cmap='gray')

        plt.savefig('love_pc.png')
        plt.close()

        plt.figure()
        plt.plot(np.cumsum(faces_pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.savefig('love_variance.png')
        plt.close()

    print('writing...')
    with open('./pca.pckl','wb') as f:
        pickle.dump(faces_pca,f)

    print(faces)
    components = faces_pca.transform(faces)


    # import pickle
    # with open('hc_single_temporal_small.pckl','rb') as f:  # Python 3: open(..., 'rb')
    # feature_allClip, feature_allClip_normal,select_info = pickle.load(f)
    # feature_allClip_normal = np.asarray(feature_allClip_normal)
    # feature_allClip = np.asarray(feature_allClip)

    # for k in range(len(components)):
    #     img_name = faces.index.values[k]
    #     frame_idx = img_name[0:-4].split('_')[-1]
    #     components[int(frame_idx)].append()
    #
    # Z = linkage(components, method ='ward',metric='euclidean')
    #
    # # print(components.shape)
    # # projected = faces_pca.inverse_transform(components)
    # # fig, axes = plt.subplots(10,10,figsize=(9,9), subplot_kw={'xticks':[], 'yticks':[]},
    # #         gridspec_kw=dict(hspace=0.01, wspace=0.01))
    # # for i, ax in enumerate(axes.flat):
    # #     ax.imshow(projected[i].reshape(200,200),cmap="gray")
    # #
    # # plt.savefig('love_ranked.png')
    #
    # f = plt.figure(figsize = (15,5))
    # g0 = plt.gca()
    # dn = hierarchy.dendrogram(Z,
    #     above_threshold_color='y',
    #     orientation='top',
    #     ax = g0)
    # plt.tight_layout()
    # plt.savefig('love_dendrogram.png')

    # # print(dn.keys())
    # leaves = np.asarray(dn['leaves'])
    # # print(leaves.size)

    # threshold = 100
    # cluster_idx = hierarchy.fcluster(Z,threshold,'distance')
    #
    # cluster_clips = [[] for i in range(np.max(cluster_idx)+1)]
    # for ii in range(len(cluster_idx)):
    #     cluster_clips[cluster_idx[ii]].append(ii)
    #
    #
    # # cluster_clips = np.asarray(cluster_clips)
    # for cluster_id in range(1,np.max(cluster_idx)+1):
    #
    #     fig, axes = plt.subplots(4,5,figsize=(9,7),
    #         subplot_kw = {'xticks':[], 'yticks':[]},
    #         gridspec_kw = dict(hspace=0.01, wspace=0.01))
    #
    #     for i, ax in enumerate(axes.flat):
    #         print(len(cluster_clips[cluster_id]))
    #         if i == len(cluster_clips[cluster_id]):
    #             break
    #         ax.imshow(faces.iloc[cluster_clips[cluster_id][i]].values.reshape(200,200),cmap='gray')
    #
    #     plt.savefig('love_cluster_{}.png'.format(cluster_id))
    #
    #     visualize_one_cluster(cluster_id,cluster_clips[cluster_id],leaves,g0,threshold,f)


if __name__ == '__main__':
    pca_for_faces()
