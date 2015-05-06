from __future__ import division

__author__ = 'tmathai'



import sys
import os.path
import glob
import cv2
import numpy as np
import timeit
import random
from random import randint
from sklearn import cross_validation, svm

import matplotlib
from matplotlib import pyplot as plt
from skimage import data, color, exposure
from skimage.feature import (
    greycomatrix, greycoprops, hog, local_binary_pattern, daisy
)

######################################################################################################################

# globals

# cropping rows and cols to be extracted
cropLeftC = 105
cropRightC = 425
cropTopR = 20
cropBotR = 365

eps = sys.float_info.epsilon

patchSize = 90
totalPatches = 200

basePatchSize = (20, 20)     # https://github.com/saurabhme/discriminative-patches/blob/384ba16da8b534aa5a04fbe96524f08cf4714ec8/code/utils/getParamsForCategory.m
levelScale = 0.5

######################################################################################################################
######################################################################################################################

def create_data_split(fullFileNameList, totalImNum):

    N1List = []
    N2List = []
    D1List = []
    D2List = []

    # copy of the file name list - items will be removed after they are randomly picked
    fList = fullFileNameList

    fLen = totalImNum

    numSplit = 45

    n1Len = int(fLen/2) - numSplit
    n2Len = int(fLen/2) - numSplit
    d1Len = int(fLen/2) - n1Len
    d2Len = int(fLen/2) - n2Len

    # randomly pick filenames for N1
    for i in range(0, n1Len):

        # randomly pick a name from the main list
        f = random.choice(list(fList))
        # add it to the list
        N1List.append(f)
        # remove it from the main list
        fList.remove(f)

    # randomly pick filenames for N2
    for i in range(0, n2Len):

        # randomly pick a name from the main list
        f = random.choice(list(fList))
        # add it to the list
        N2List.append(f)
        # remove it from the main list
        fList.remove(f)

    # randomly pick filenames for D1
    for i in range(0, d1Len):

        # randomly pick a name from the main list
        f = random.choice(list(fList))
        # add it to the list
        D1List.append(f)
        # remove it from the main list
        fList.remove(f)

    # assign the remaining filenames to D2
    # randomly pick filenames for D1
    for i in range(0, d2Len):

        # randomly pick a name from the main list
        f = random.choice(list(fList))
        # add it to the list
        D2List.append(f)
        # remove it from the main list
        fList.remove(f)

    return D1List, N1List, D2List, N2List

######################################################################################################################

def getGradient(aIm):

    sobelx = cv2.Sobel(aIm, cv2.CV_64F, 1,0, ksize=-1)
    sobely = cv2.Sobel(aIm, cv2.CV_64F, 0,1, ksize=-1)

    sumSobel = cv2.add(sobelx, sobely)

    sqSobel = 2*np.array(sumSobel)

    return sqSobel

######################################################################################################################

def getProbDistribution(img1, pSize):

    # get the gaussian filter
    # blur = cv2.GaussianBlur(img1,(5,5),levelScale)
    blur = cv2.GaussianBlur(img1,pSize,levelScale)

    totalSum = np.sum(blur)
    dist = np.array(img1)/ totalSum
    return dist

######################################################################################################################

def sampleRandomImagePatches(im):

    img_patches1 = dict()
    grad_patches1 = dict()

    # compute the horiz and vert gradients of the image
    sqSobel = getGradient(im)

    # not used
    # pSize = (levelScale * np.array(basePatchSize))  # multiply patch size by scale
    # pSize = np.floor(pSize)     # floor values
    # pSize = pSize.astype(int)   # convert to integer
    # pDist = getProbDistribution(sqSobel, pSize)

    [rr,cc] = im.shape

    featvec, hogIm = hog(im,
                  orientations=8,
                  pixels_per_cell=(6, 6),
                  cells_per_block=(8, 8),
                  visualise=True,
                  normalise=True)

    maxEnergy1 = []

    # here we extract patches randomly from the image
    #  patches can have low energy
    for i in range(0,totalPatches):
        # get random row and column indicies
        rowI = randint(0, rr-patchSize-1)
        colI = randint(0, cc-patchSize-1)

        # extract patchSize square patch from original image
        img_patches1[i] = im[rowI : rowI+patchSize, colI : colI+patchSize]

        # extract patchSize square patch from gradient image
        grad_patches1[i] = hogIm[rowI : rowI+patchSize, colI : colI+patchSize]

        hogpSum = np.sum(grad_patches1[i])
        hogpAvg = hogpSum/ (patchSize*patchSize)

        maxEnergy1.append(hogpAvg)

    # maxIndex = np.argmax(maxEnergy1)
    # minIndex = np.argmin(maxEnergy1)
    #
    # maxVal = maxEnergy1[maxIndex]
    # minVal = maxEnergy1[minIndex]
    #
    # print maxVal, minVal

    #  compute a threshold
    # take 85% of the energy value as the threshold
    # this serves to remove patches whose energies are less than this threshold
    thresh = (85*np.mean(maxEnergy1))/100

    # discard the patches with low energy with previously computed threshold
    for i in range(0,totalPatches):

        if(maxEnergy1[i]<thresh):   # check if energy is less than computed threshold
            while(maxEnergy1[i]<thresh):    # while loop continues till maxEnergy[i] is greater than thresh

                # resample the image to get a new patch (potentially with higher energy)
                # get random row and column indicies
                rowI = randint(0, rr-patchSize)
                colI = randint(0, cc-patchSize)

                # original image - extract patchSize square patch
                img_patches1[i] = im[rowI : rowI+patchSize, colI : colI+patchSize]

                # gradient image - extract patchSize square patch
                grad_patches1[i] = hogIm[rowI : rowI+patchSize, colI : colI+patchSize]

                hogpSum = np.sum(grad_patches1[i])
                maxEnergy1[i] = hogpSum/ (patchSize*patchSize)

    # featvec2 = np.histogram(featvec, bins=8)

    # take the first patch - convert to row vector
    # the samples matrix input to KMEANS must be of type float32
    samplesMatrix1 = np.float32(grad_patches1[0].reshape((1,-1)))
    imRowMatrix1 = img_patches1[0].reshape((1,-1))

    # store all patches (num_Patches x N x N) in matrix with rows as samples -- ( num_Patches x (NxN) )
    for i in range(1,totalPatches):

        # convert to row vector and stack vertically on top of each other
        rowReshape = np.float32(grad_patches1[i].reshape((1,-1)))
        imRowPatch = img_patches1[i].reshape((1,-1))

        samplesMatrix1  = np.vstack((samplesMatrix1,rowReshape))
        imRowMatrix1 = np.vstack((imRowMatrix1,imRowPatch))



    return imRowMatrix1, samplesMatrix1, maxEnergy1

########################################################################################################################

def get_stacked_patches(dataConsidered1, fullPath1):

    fullKMEANMatrix1 = ()
    imageMatrix1 = ()

    for fileNameIndex1 in range(0, len(dataConsidered1)):

        print "Reading image:", dataConsidered1[fileNameIndex1]
        img1 = cv2.imread(os.path.join(fullPath1, dataConsidered1[fileNameIndex1]))
        gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # crop the image
        # similar to matlab, extract as (rows,cols)
        # cropImg1 = img1[cropTopR:cropBotR, cropLeftC:cropRightC]
        cropGrayImg1 = gimg1[cropTopR:cropBotR, cropLeftC:cropRightC]

        # cv2.imwrite(os.path.join(croppedPath, fullFileNameList[imCount]), cropImg1)

        img_patches2, samplesMatrix2, maxEnergy2 = sampleRandomImagePatches(cropGrayImg1)

        if(fileNameIndex1 == 0):
            fullKMEANMatrix1 = samplesMatrix2
            imageMatrix1 = img_patches2
        else:
            fullKMEANMatrix1 = np.vstack((fullKMEANMatrix1,samplesMatrix2))
            imageMatrix1 = np.vstack((imageMatrix1,img_patches2))

    return imageMatrix1, fullKMEANMatrix1


########################################################################################################################
########################################################################################################################

if __name__ == "__main__":


    fullPath = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/images"
    croppedPath = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/croppedImages"

    outputPath = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/output"
    iter1Path = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/iter1_output"
    iter2Path = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/iter2_output"

    fullFileNameList = []

    imCount = 0

    start = timeit.default_timer()

    print "\n", "################## Read images ##################"

    print "Reading images from folder:", fullPath

    # read the filenames of all the input images
    for file in os.listdir(fullPath):
        #print "Reading image:", file

        # read the image and add it to the list of images
        fullFileNameList.append(file)

        img = cv2.imread(os.path.join(fullPath, fullFileNameList[imCount]))
        cropImg = img[cropTopR:cropBotR, cropLeftC:cropRightC]
        cv2.imwrite(os.path.join(croppedPath, fullFileNameList[imCount]), cropImg)

        imCount += 1

    # get the number of images in the directory
    totalImNum = len(fullFileNameList)

##############################################################################

    print "\n", "################## Creating data split ##################"

    print "\n", "Number of input files:", len(fullFileNameList)

    # get the data split
    D1, N1, D2, N2 = create_data_split(fullFileNameList, totalImNum)

    print "\n", "size of dataset: ", len(D1), len(D2), len(N1), len(N2)

    print "\n", "Done with data split."

##############################################################################


    print "\n", "################## Sampling random patches from D1 ##################"

    print "\n", "Length of D1", len(D1)

    imPatchesD1, stackPatchesD1 = get_stacked_patches(D1, fullPath)
    [rd1,cd1] = stackPatchesD1.shape

##############################################################################

    print "\n", "################## Sampling random patches from D2 ##################"
    print "\n", "Length of D2", len(D2)

    imPatchesD2, stackPatchesD2 = get_stacked_patches(D2, fullPath)
    [rd2,cd2] = stackPatchesD2.shape

##############################################################################

    print "\n", "################## Sampling random patches from N1 ##################"
    print "\n", "Length of N1", len(N1)

    imPatchesN1, stackPatchesN1 = get_stacked_patches(N1, fullPath)
    [rn1,cn1] = stackPatchesN1.shape

##############################################################################

    print "\n", "################## Sampling random patches from N2 ##################"
    print "\n", "Length of N2", len(N2)

    imPatchesN2, stackPatchesN2 = get_stacked_patches(N2, fullPath)
    [rn2,cn2] = stackPatchesN2.shape

##############################################################################

    print "\n", "################## Computing KMEANS for D1 ##################"

    # intial kmeans clusters should be S/4
    num_clustCenters = int(rd1/4)

    print "\n", "num_clusters", num_clustCenters

    start_time = timeit.default_timer()

    # define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, eps)

    # apply kmeans()
    ret,label,center = cv2.kmeans(stackPatchesD1,num_clustCenters,criteria,15,cv2.KMEANS_RANDOM_CENTERS)

    elapsed = timeit.default_timer() - start_time

    print "\n", "Elapsed Time for D1 KMEANS:", elapsed

##############################################################################

    print "\n", "################## Initialize K ##################"

    K = {}
    Knew = {}
    imPK = {}
    imPKnew = {}
    pIndK = []
    pLenK = []

    # number of patches per cluster
    mPatchPerClus = 4

    ki = 0
    # store the clusters in initial K dictionary
    for ci in range(0,num_clustCenters):

        # get patches for a cluster index
        temp = stackPatchesD1[label.ravel() == ci]
        [rk,ck] = temp.shape
        # print rk, ck

        # store indicies of clusters with patch counts that are greater than 4
        if(rk >= mPatchPerClus):

            pIndK.append(ci)    # store cluster index
            pLenK.append(rk)    # store num_patches in cluster

            # extract the patches of a cluster from D1, and store in K
            K[ki] = stackPatchesD1[label.ravel() == ci]

            ki += 1

    print "\n", "pruned num_clusters", len(K)

##############################################################################

    print "\n", "################## Convergence of patch algorithm ##################"

    clusPurity = []

    discrim = []

    # now apply the convergence algorithm
    for iter in range(1,8):

        if (iter % 2 == 0):
            print "even iter:", iter
            trainingSet = stackPatchesN2
            testingSet = stackPatchesD1
            testingImageSet = imPatchesD1
            print "Training with negative N2 samples, and validating with D1"
        else:
            print "even iter:", iter
            trainingSet = stackPatchesN1
            testingSet = stackPatchesD2
            testingImageSet = imPatchesD2
            print "Training with negative N1 samples, and validating with D2"

        # get number of clusters
        numClusts = len(K)
        print "numClusts:", numClusts

        cnt = 0

        for i in range(0,numClusts):

            #####################################
            # training stage

            print "training stage - iter:", iter, " cluster:", i

            # generate positive binary labels for the patches in the current D1 cluster
            [rp,cp] = K[i].shape
            pLabel = [1] * rp

            # generate negative binary labels for all the trainingSet patches
            [rn,cn] = trainingSet.shape
            nLabel = [0] * rn

            # add the values from the two labels together
            fullLabel = pLabel + nLabel

            # extract the positive patches in the current cluster
            posPatches = K[i]

            # vertically stack the positive patches on top of the negative patches
            feats = np.vstack((posPatches,trainingSet))

            # start the time for training
            start_time = timeit.default_timer()

            # train the linear SVM classifier on the pos_patches and neg_patches
            clf = svm.SVC(kernel='linear', probability=True)
            clf.fit(feats, fullLabel)

            #####################################
            # testing stage

            print "testing stage - iter:", iter, " cluster:", i

            result = clf.predict(testingSet)

            probs = clf.predict_log_proba(testingSet)

            # print training time for ith classifier
            elapsed = timeit.default_timer() - start_time

            print "Elapsed time for training and testing classifier on cluster ", i, " :", elapsed

            print "out of", len(result), "patches, non-zero patches:", np.count_nonzero(result)

            # column 1 corresponds to positive classes -- clf.classes_ == [0,1]
            probs1 = probs[:,1]

            if(iter > 2):

                if len(clusPurity) != 0:
                    clusPurity = []

                if len(discrim) != 0:
                    discrim = []

                # calculate cluster purity
                # sort the positive probabilities in ascending order
                sortProbs1 = np.sort(probs1)

                # extract the last mPatchesPerClus values from the sorted prob vector
                # get the sum value
                sumProbs = np.sum(sortProbs1[(len(sortProbs1)-mPatchPerClus):(len(sortProbs1))])

                clusPurity.append(sumProbs)

                # calculate cluster discriminativeness
                # calculate results on all D_patches
                resultfD = clf.predict(np.vstack((stackPatchesD1,stackPatchesD2)))
                nzrD = np.count_nonzero(resultfD)

                # calculate cluster discriminativeness
                # calculate results on all N_patches
                resultfDN = clf.predict( np.vstack(( np.vstack((stackPatchesD1,stackPatchesD2)) , np.vstack((stackPatchesN1,stackPatchesN2)) )) )
                nzrDN = np.count_nonzero(resultfDN)

                vtmp = nzrD/nzrDN

                discrim.append(vtmp)

            # this gets the indicies of the probs1 elements sorted in ascending order
            # we need the values which have the high probabilites - towards the end
            probInd = np.argsort(probs1)

            ind = []

            # extract the last five indicies of the sorted probs
            if(np.count_nonzero(result) >= mPatchPerClus):

                for ei in range(len(probInd)-mPatchPerClus,len(probInd)):
                    ind.append(probInd[ei])

                print "Collecting new patches"

                # if the testingSet cluster contains at minimum n members, we take them
                # consistent with paper - if detector fires less than 2 times on validation set, we kill the cluster
                if(len(ind) >= mPatchPerClus):
                    Knew[cnt] = testingSet[ind,:]
                    imPKnew[cnt] = testingImageSet[ind,:]
                    print "Len Knew[cnt]:", len(Knew[cnt])
                    cnt += 1

            # ind = []
            #
            # mPatchPerClus = 4
            #
            # if(np.count_nonzero(result) >= mPatchPerClus):
            #     for ei, el in enumerate(result):
            #         if(el > 0):
            #             ind.append(ei)
            #
            #         if(len(ind) == mPatchPerClus):
            #             break
            #
            #     # print "lenInd:", len(ind)
            #     print "Collecting new patches"
            #
            #     # if the testingSet cluster contains at minimum 5 members, we take them
            #     # consistent with paper - if detector fires less than 2 times on validation set, we kill the cluster
            #     if(len(ind) >= mPatchPerClus):
            #         Knew[cnt] = testingSet[ind,:]
            #         imPKnew[cnt] = testingImageSet[ind,:]
            #         print "Len Knew[cnt]:", len(Knew[cnt])
            #         cnt += 1

        if(iter <= 2):

            print "Not converged yet."
            # replace old K with new Knew obtained after testing with D2
            print "\n", "old K:", len(K), "new Knew:", len(Knew)
            K = Knew
            imPK = imPKnew

            # write iteration 1 results to disk
            if(iter == 1):

                for the_file in os.listdir(iter1Path):
                    file_path = os.path.join(iter1Path, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                    except Exception, e:
                        print e

                # now save the clustering results to disk
                for i in range(0,len(K)):
                    temp = imPK[i]
                    [rr,cc] = temp.shape
                    for p in range(0,rr):
                        patU = temp[p,:]
                        cv2.imwrite(os.path.join(iter1Path, "clus" + str(i) + "_" + str(p) + ".png"), patU.reshape(patchSize,patchSize))

            # write iteration 2 results to disk
            else:

                for the_file in os.listdir(iter2Path):
                    file_path = os.path.join(iter2Path, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                    except Exception, e:
                        print e

                # now save the clustering results to disk
                for i in range(0,len(K)):
                    temp = imPK[i]
                    [rr,cc] = temp.shape
                    for p in range(0,rr):
                        patU = temp[p,:]
                        cv2.imwrite(os.path.join(iter2Path, "clus" + str(i) + "_" + str(p) + ".png"), patU.reshape(patchSize,patchSize))

        # check if the results from the previous iteration are the same as results of current iteration
        # if true, then save all current results to disk
        if(iter > 2):

            overallKCount = 0
            for i in range(0,numClusts):
                kcount = 0
                for k in range(0,len(Knew[i])):
                    corrVal = np.corrcoef( [K[i][k], Knew[i][k]] )[0,1]   #correlation of array at row-0, col-1
                    if(corrVal == 1.0):
                        kcount += 1
                if(int(kcount) == mPatchPerClus):
                    overallKCount += 1
                print "overallKCount for cluster", i+1,"=",overallKCount

            if(overallKCount == numClusts):
                print "Converged. Exiting loop, and writing results."
                break
            else:
                # replace old K with new Knew obtained after testing with D2
                print "Not converged yet."
                print "\n", "old K:", len(K), "new Knew:", len(Knew)
                K = Knew
                imPK = imPKnew


    print "\n", "Writing patches to disk"

    print "\n", "Deleting existing patches in folder"

    for the_file in os.listdir(outputPath):
        file_path = os.path.join(outputPath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception, e:
            print e

    # now save the final clustering results to disk
    for i in range(0,len(K)):
        temp = imPK[i]
        [rr,cc] = temp.shape
        for p in range(0,rr):
            patU = temp[p,:]
            cv2.imwrite(os.path.join(outputPath, "clus" + str(i) + "_" + str(p) + ".png"), patU.reshape(patchSize,patchSize))


    # display the graphs for cluster purity and the cluster discriminativeness
    plt.figure(0)
    plt.plot(np.arange(0,len(clusPurity)), clusPurity)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cluster Purity')
    plt.show()

    plt.figure(1)
    plt.plot(np.arange(0,len(discrim)), discrim)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Discriminativeness')
    plt.show()

    stop = timeit.default_timer()
    fpTime = stop - start
    print "Full program runtime:", fpTime/60, "minutes"



# ##############################################################################
#
#     print "\n", "################## Pruning D1 KMEANS ##################"
#
#     # store indicies of D1 clusters which have greater than 4 members
#     pruned_gIndD1 = []
#
#     # store length of D1 clusters which have greater than 4 members
#     pruned_gLenD1 = []
#
#     # store D1 clusters which have greater than 4 members
#     # pruned_gClusD1 = np.array([])
#     # pruned_gClusD1.shape=(0,cc)
#     pruned_gClusD1 = {}
#
#     count = 0
#     # store the indicies of the clusters with more than 3 patches
#     for i in range(0,num_clustCenters):
#         temp = stackPatchesD1[label.ravel() == i]
#         [rt,ct] = temp.shape
#         # print rt, ct
#
#         if(rt > 4):
#             pruned_gIndD1.append(i)
#             pruned_gLenD1.append(rt)
#             # pruned_gClusD1 = np.vstack( (pruned_gClusD1,temp) )
#             pruned_gClusD1[count] = temp
#             count += 1
#
#     # print "\n", "num_clusters", num_clustCenters, "len pruned_gIndD1", len(pruned_gIndD1)
#     prunedClusterCountD1 = (len(pruned_gIndD1))
#     # print len(pruned_gIndD1)
#     # print (pruned_gLenD1[0])
#     # print pruned_gClusD1[0].shape[0]
#     print "\n", "Removed cluster percentage for D1: ", float((num_clustCenters - prunedClusterCountD1)/num_clustCenters)
#
#
#
# ##############################################################################
#
#     print "\n", "Training on held-out N1 as all-negative examples", "\n"
#
#     # generate negative binary labels for all the N1 patches
#     nLabelN1 = [0] * rn1
#
#     clfStore = {}
#
#     for i in range(0,1):
#
#         # generate positive binary labels for the patches in the current D1 cluster
#         pLabelD1 = [1] * pruned_gLenD1[i]
#
#         # add the values from the two labels together
#         fullLabel = pLabelD1 + nLabelN1
#
#         # extract the positive D1 patches in the current cluster
#         d1patches = pruned_gClusD1[i]
#         [rr,cc] = d1patches.shape
#         print "d1: ", rr, pruned_gLenD1[i], "n1: ", rn1, "d1 + n1: ", rr + rn1, "labLen: ", len(fullLabel)
#
#         # vertically stack the positive cluster with D1 patches, over the negative cluster with N1 patches
#         feats = np.vstack((d1patches,stackPatchesN1))
#
#         start_time = timeit.default_timer()
#
#         # train the linear SVM classifier on the pos_D1_patches and neg_N1_patches
#         clf = svm.SVC(kernel='linear', probability=True)
#         clf.fit(feats, fullLabel)
#
#         result = clf.predict(stackPatchesD2)
#
#         # print training time for ith classifier
#         elapsed = timeit.default_timer() - start_time
#
#         print "Elapsed time for training classifier on cluster ", i, " :", elapsed
#
#         ind = []
#         cnt = 0
#         for i, el in enumerate(result):
#             if(el != 0 and cnt < 5):
#                 ind.append(i)
#                 cnt += 1
#
#         newPatchClus = stackPatchesD2[ind,:]
#
#         print newPatchClus.shape
#
#         # j2 =  [r for r in result != 0]
#         #
#         # outval = np.extract(j2, result)
#         #
#         # print "vals not zero: ", outval
#         # print "len outVal: ", len(outval)
#         #
#         # correct = np.count_nonzero(result)
#         #
#         # print "number of positives: ", correct
#
#         # clear the label in the end
#         fullLabel = []




##############################################################################




##############################################################################
    cv2.waitKey()

    cv2.destroyAllWindows()






########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


    # # code to crop the video image
     # for fileNameIndex in range(0, totalImNum):
     #
     #    print "Reading image from fullImages:", fullFileNameList[fileNameIndex]
     #    img = cv2.imread(os.path.join(fullPath, fullFileNameList[fileNameIndex]))
     #
     #    [rows,cols,depth] = img.shape
     #    print rows
     #
     #    if(rows == 480 and cols == 640):
     #
     #        # crop the image
     #        # similar to matlab, extract as (rows,cols)
     #        cropImg = img[(35-10):rows-27, (180-105):cols-100]
     #
     #        cv2.imwrite(os.path.join(fullPath, fullFileNameList[fileNameIndex]), cropImg)

    # # matlab code to read the images from a folder and save them to disk
    # source='Stationary-film-clip2.mp4';
    # vidobj=VideoReader(source);
    # frames=vidobj.Numberofframes;
    # for f=1:4:frames
    #   thisframe=read(vidobj,f);
    #   %figure(1);imagesc(thisframe);
    #   thisfile=sprintf('vid4_%d.png',f);
    #   imwrite(thisframe,thisfile);
    # end












































########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
# old copy of code

#
# from __future__ import division
#
# __author__ = 'tmathai'
#
#
#
# import sys
# import os.path
# import glob
# import cv2
# import numpy as np
# import timeit
# from random import randint
# from sklearn import cross_validation, svm
#
# from matplotlib import pyplot as plt
# from skimage import data, color, exposure
# from skimage.feature import (
#     greycomatrix, greycoprops, hog, local_binary_pattern, daisy
# )
#
# ######################################################################################################################
#
# # globals
#
# # cropping rows and cols to be extracted
# cropLeftC = 105
# cropRightC = 425
# cropTopR = 20
# cropBotR = 365
#
# eps = sys.float_info.epsilon
#
# patchSize = 90
# totalPatches = 150
#
# basePatchSize = (20, 20)     # https://github.com/saurabhme/discriminative-patches/blob/384ba16da8b534aa5a04fbe96524f08cf4714ec8/code/utils/getParamsForCategory.m
# levelScale = 0.5
#
# ######################################################################################################################
# ######################################################################################################################
#
# def create_data_split(fullFileNameList, totalImNum):
#
#     N1List = []
#     N2List = []
#     D1List = []
#     D2List = []
#
#     for i in range(0, totalImNum):
#
#         # take the first quarter of the images, and store it in D1
#         if(i < totalImNum/4):
#             file1 = fullFileNameList[i]
#             D1List.append(file1)
#
#         # take the second quarter of the images, and store it in N1
#         if( (i >= totalImNum/4)  and  (i < totalImNum/2) ):
#             file2 = fullFileNameList[i]
#             N1List.append(file2)
#
#         # take the third quarter of the images, and store it in D2
#         if( (i >= totalImNum/2)  and  (i < (3*totalImNum)/4) ):
#             file3 = fullFileNameList[i]
#             D2List.append(file3)
#
#         # take the final quarter of the images, and store it in N2
#         if( (i >= (3*totalImNum)/4)  and  (i < totalImNum) ):
#             file4 = fullFileNameList[i]
#             N2List.append(file4)
#
#     return D1List, N1List, D2List, N2List
#
# ######################################################################################################################
#
# def getGradient(aIm):
#
#     sobelx = cv2.Sobel(aIm, cv2.CV_64F, 1,0, ksize=-1)
#     sobely = cv2.Sobel(aIm, cv2.CV_64F, 0,1, ksize=-1)
#
#     sumSobel = cv2.add(sobelx, sobely)
#
#     sqSobel = 2*np.array(sumSobel)
#
#     # abs_sobelx64f = np.absolute(sobelx)
#     # sobelx_8u = np.uint8(abs_sobelx64f)
#     #
#     # abs_sobely64f = np.absolute(sobely)
#     # sobely_8u = np.uint8(abs_sobely64f)
#
#     # return sobelx, sobely
#
#     return sqSobel
#
# ######################################################################################################################
#
# def getProbDistribution(img1, pSize):
#
#     # get the gaussian filter
#     # blur = cv2.GaussianBlur(img1,(5,5),levelScale)
#     blur = cv2.GaussianBlur(img1,pSize,levelScale)
#
#     totalSum = np.sum(blur)
#     dist = np.array(img1)/ totalSum
#     return dist
#
# ######################################################################################################################
#
# def sampleRandomImagePatches(im):
#
#     img_patches1 = dict()
#     grad_patches1 = dict()
#
#     # compute the horiz and vert gradients of the image
#     sqSobel = getGradient(im)
#
#     # not used
#     # pSize = (levelScale * np.array(basePatchSize))  # multiply patch size by scale
#     # pSize = np.floor(pSize)     # floor values
#     # pSize = pSize.astype(int)   # convert to integer
#     # pDist = getProbDistribution(sqSobel, pSize)
#
#     [rr,cc] = im.shape
#
#     featvec, hogIm = hog(im,
#                   orientations=8,
#                   pixels_per_cell=(6, 6),
#                   cells_per_block=(8, 8),
#                   visualise=True,
#                   normalise=True)
#
#     maxEnergy1 = []
#
#     # here we extract patches randomly from the image
#     #  patches can have low energy
#     for i in range(0,totalPatches):
#         # get random row and column indicies
#         rowI = randint(0, rr-patchSize)
#         colI = randint(0, cc-patchSize)
#
#         # extract patchSize square patch from original image
#         img_patches1[i] = im[rowI : rowI+patchSize, colI : colI+patchSize]
#
#         # extract patchSize square patch from gradient image
#         grad_patches1[i] = hogIm[rowI : rowI+patchSize, colI : colI+patchSize]
#
#         hogpSum = np.sum(grad_patches1[i])
#         hogpAvg = hogpSum/ (patchSize*patchSize)
#
#         maxEnergy1.append(hogpAvg)
#
#     # maxIndex = np.argmax(maxEnergy1)
#     # minIndex = np.argmin(maxEnergy1)
#     #
#     # maxVal = maxEnergy1[maxIndex]
#     # minVal = maxEnergy1[minIndex]
#     #
#     # print maxVal, minVal
#
#     #  compute a threshold
#     # take 85% of the energy value as the threshold
#     # this serves to remove patches whose energies are less than this threshold
#     thresh = (85*np.mean(maxEnergy1))/100
#
#     # discard the patches with low energy with previously computed threshold
#     for i in range(0,totalPatches):
#
#         if(maxEnergy1[i]<thresh):   # check if energy is less than computed threshold
#             while(maxEnergy1[i]<thresh):    # while loop continues till maxEnergy[i] is greater than thresh
#
#                 # resample the image to get a new patch (potentially with higher energy)
#                 # get random row and column indicies
#                 rowI = randint(0, rr-patchSize)
#                 colI = randint(0, cc-patchSize)
#
#                 # original image - extract patchSize square patch
#                 img_patches1[i] = im[rowI : rowI+patchSize, colI : colI+patchSize]
#
#                 # gradient image - extract patchSize square patch
#                 grad_patches1[i] = hogIm[rowI : rowI+patchSize, colI : colI+patchSize]
#
#                 hogpSum = np.sum(grad_patches1[i])
#                 maxEnergy1[i] = hogpSum/ (patchSize*patchSize)
#
#     # featvec2 = np.histogram(featvec, bins=8)
#
#     # take the first patch - convert to row vector
#     # the samples matrix input to KMEANS must be of type float32
#     samplesMatrix1 = np.float32(grad_patches1[0].reshape((1,-1)))
#     imRowMatrix1 = img_patches1[0].reshape((1,-1))
#
#     # store all patches (num_Patches x N x N) in matrix with rows as samples -- ( num_Patches x (NxN) )
#     for i in range(1,totalPatches):
#
#         # convert to row vector and stack vertically on top of each other
#         rowReshape = np.float32(grad_patches1[i].reshape((1,-1)))
#         imRowPatch = img_patches1[i].reshape((1,-1))
#
#         samplesMatrix1  = np.vstack((samplesMatrix1,rowReshape))
#         imRowMatrix1 = np.vstack((imRowMatrix1,imRowPatch))
#
#
#
#     return imRowMatrix1, samplesMatrix1, maxEnergy1
#
# ########################################################################################################################
#
# def get_stacked_patches(dataConsidered1, fullPath1):
#
#     fullKMEANMatrix1 = ()
#     imageMatrix1 = ()
#
#     for fileNameIndex1 in range(0, len(dataConsidered1)):
#
#         print "Reading image:", dataConsidered1[fileNameIndex1]
#         img1 = cv2.imread(os.path.join(fullPath1, dataConsidered1[fileNameIndex1]))
#         gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#
#         # crop the image
#         # similar to matlab, extract as (rows,cols)
#         # cropImg1 = img1[cropTopR:cropBotR, cropLeftC:cropRightC]
#         cropGrayImg1 = gimg1[cropTopR:cropBotR, cropLeftC:cropRightC]
#
#         # cv2.imwrite(os.path.join(croppedPath, fullFileNameList[imCount]), cropImg1)
#
#         img_patches2, samplesMatrix2, maxEnergy2 = sampleRandomImagePatches(cropGrayImg1)
#
#         if(fileNameIndex1 == 0):
#             fullKMEANMatrix1 = samplesMatrix2
#             imageMatrix1 = img_patches2
#         else:
#             fullKMEANMatrix1 = np.vstack((fullKMEANMatrix1,samplesMatrix2))
#             imageMatrix1 = np.vstack((imageMatrix1,img_patches2))
#
#     return imageMatrix1, fullKMEANMatrix1
#
#
# ########################################################################################################################
# ########################################################################################################################
#
# if __name__ == "__main__":
#
#
#     fullPath = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/images"
#     croppedPath = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/croppedImages"
#     outputPath = "C:/Users/tmathai/Dropbox/RI/Sem 2/LBMV/Project/testset/output"
#
#     fullFileNameList = []
#
#     imCount = 0
#
#     start = timeit.default_timer()
#
#     print "Reading images from folder:", fullPath
#
#     # read the filenames of all the input images
#     for file in os.listdir(fullPath):
#         #print "Reading image:", file
#
#         # read the image and add it to the list of images
#         fullFileNameList.append(file)
#
#         img = cv2.imread(os.path.join(fullPath, fullFileNameList[imCount]))
#         cropImg = img[cropTopR:cropBotR, cropLeftC:cropRightC]
#         cv2.imwrite(os.path.join(croppedPath, fullFileNameList[imCount]), cropImg)
#
#         imCount += 1
#
#     # get the number of images in the directory
#     totalImNum = len(fullFileNameList)
#
# ##############################################################################
#
#     print "\n", "Creating data split:", "\n"
#
#     # get the data split
#     D1, N1, D2, N2 = create_data_split(fullFileNameList, totalImNum)
#
#     print "\n", "size of dataset: ", len(D1), len(D2), len(N1), len(N2)
#
#     print "\n", "Done with data split.", "\n"
#
# ##############################################################################
#
#     print "\n", "Length of D1", len(D1)
#     print "\n", "Sampling random patches from D1", "\n"
#
#     imPatchesD1, stackPatchesD1 = get_stacked_patches(D1, fullPath)
#     [rd1,cd1] = stackPatchesD1.shape
#     # [rid1,cid1] = imPatchesD1.shape
#     #
#     # print rd1, rid1
#
#     # pat = stackPatchesD1[1,:].reshape(patchSize,patchSize)
#     # cv2.imshow('dst_rt', pat)
#     # maxp = np.amax(pat)
#     # patF = pat/maxp
#     # patU = pat * 255
#     # cv2.imwrite(os.path.join(outputPath,("dst_rt" + ".png")), patU)
#
#     # for i in range(0,1):
#     #     temp = stackPatchesD1
#     #     [rr,cc] = temp.shape
#     #     for p in range(0,rr):
#     #         patU = temp[p,:] * 255
#     #         cv2.imwrite(os.path.join(outputPath, "clus" + str(i) + "_" + str(p) + ".png"), patU.reshape(patchSize,patchSize))
#
# ##############################################################################
#
#     print "\n", "Length of D2", len(D2)
#     print "\n", "Sampling random patches from D2", "\n"
#
#     imPatchesD2, stackPatchesD2 = get_stacked_patches(D2, fullPath)
#     [rd2,cd2] = stackPatchesD2.shape
#     # [rid2,cid2] = imPatchesD2.shape
#     #
#     # print rd2, rid2
#
# ##############################################################################
#
#     print "\n", "Length of N1", len(N1)
#     print "\n", "Sampling random patches from N1", "\n"
#
#     imPatchesN1, stackPatchesN1 = get_stacked_patches(N1, fullPath)
#     [rn1,cn1] = stackPatchesN1.shape
#     # [rin1,cin1] = imPatchesN1.shape
#     #
#     # print rn1, rin1
#
# ##############################################################################
#
#     print "\n", "Length of N2", len(N2)
#     print "\n", "Sampling random patches from N2", "\n"
#
#     imPatchesN2, stackPatchesN2 = get_stacked_patches(N2, fullPath)
#     [rn2,cn2] = stackPatchesN2.shape
#     # [rin2,cin2] = imPatchesN2.shape
#     #
#     # print rn2, rin2
#
# ##############################################################################
#
#     print "\n", "Computing KMEANS for D1"
#
#     # intial kmeans clusters should be S/4
#     num_clustCenters = int(rd1/50)
#
#     print "\n", "num_clusters", num_clustCenters
#
#     start_time = timeit.default_timer()
#
#     # define criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, eps)
#
#     # apply kmeans()
#     ret,label,center = cv2.kmeans(stackPatchesD1,num_clustCenters,criteria,15,cv2.KMEANS_RANDOM_CENTERS)
#
#     elapsed = timeit.default_timer() - start_time
#
#     print "\n", "Elapsed Time for D1 KMEANS:", elapsed
#
# ##############################################################################
#
#     print "\n", "Initialize K"
#
#     K = {}
#     Knew = {}
#     imPK = {}
#     imPKnew = {}
#     pIndK = []
#     pLenK = []
#
#     ki = 0
#     # store the clusters in initial K dictionary
#     for ci in range(0,num_clustCenters):
#
#         # get patches for a cluster index
#         temp = stackPatchesD1[label.ravel() == ci]
#         [rk,ck] = temp.shape
#         # print rk, ck
#
#         # store indicies of clusters with patch counts that are greater than 4
#         if(rk > 4):
#
#             pIndK.append(ci)    # store cluster index
#             pLenK.append(rk)    # store num_patches in cluster
#
#             # extract the patches of a cluster from D1, and store in K
#             K[ki] = stackPatchesD1[label.ravel() == ci]
#
#             ki += 1
#
#     print "\n", "pruned num_clusters", len(K)
#
# ##############################################################################
#
#     print "\n", "Convergence of patch algorithm", "\n"
#
#     # now apply the convergence algorithm
#     for iter in range(1,6):
#
#         if (iter % 2 == 0):
#             print "even iter:", iter
#             trainingSet = stackPatchesN2
#             testingSet = stackPatchesD1
#             testingImageSet = imPatchesD1
#         else:
#             print "even iter:", iter
#             trainingSet = stackPatchesN1
#             testingSet = stackPatchesD2
#             testingImageSet = imPatchesD2
#
#         # get number of clusters
#         numClusts = len(K)
#         print "numClusts:", numClusts
#
#         cnt = 0
#
#         for i in range(0,numClusts):
#
#             #####################################
#             # training stage
#
#             print "training stage - iter:", iter, " cluster:", i
#
#             # generate positive binary labels for the patches in the current D1 cluster
#             [rp,cp] = K[i].shape
#             pLabel = [1] * rp
#
#             # generate negative binary labels for all the trainingSet patches
#             [rn,cn] = trainingSet.shape
#             nLabel = [0] * rn
#
#             # add the values from the two labels together
#             fullLabel = pLabel + nLabel
#
#             # extract the positive patches in the current cluster
#             posPatches = K[i]
#
#             # vertically stack the positive patches on top of the negative patches
#             feats = np.vstack((posPatches,trainingSet))
#
#             # start the time for training
#             start_time = timeit.default_timer()
#
#             # train the linear SVM classifier on the pos_patches and neg_patches
#             clf = svm.SVC(kernel='linear', probability=True)
#             clf.fit(feats, fullLabel)
#
#             #####################################
#             # testing stage
#
#             print "testing stage - iter:", iter, " cluster:", i
#
#             result = clf.predict(testingSet)
#
#             # print training time for ith classifier
#             elapsed = timeit.default_timer() - start_time
#
#             print "Elapsed time for training and testing classifier on cluster ", i, " :", elapsed
#
#             print "out of", len(result), "patches, non-zero patches:", np.count_nonzero(result)
#
#             ind = []
#
#             mPatchPerClus = 4
#
#             if(np.count_nonzero(result) >= mPatchPerClus):
#                 for ei, el in enumerate(result):
#                     if(el > 0):
#                         ind.append(ei)
#
#                     if(len(ind) == mPatchPerClus):
#                         break
#
#                 print "lenInd:", len(ind)
#                 print "Collecting new patches"
#
#                 # if the testingSet cluster contains at minimum 5 members, we take them
#                 # consistent with paper - if detector fires less than 2 times on validation set, we kill the cluster
#                 if(len(ind) >= mPatchPerClus):
#                     Knew[cnt] = testingSet[ind,:]
#                     imPKnew[cnt] = testingImageSet[ind,:]
#                     print "Len Knew[cnt]:", len(Knew[cnt])
#                     cnt += 1
#
#         # replace old K with new Knew obtained after testing with D2
#         print "old K:", len(K), "new Knew:", len(Knew)
#         K = Knew
#         imPK = imPKnew
#
#     print "\n", "Writing patches to disk"
#
#     # now save the final clustering results to disk
#     for i in range(0,len(K)):
#         temp = imPK[i]
#         [rr,cc] = temp.shape
#         for p in range(0,rr):
#             patU = temp[p,:]
#             cv2.imwrite(os.path.join(outputPath, "clus" + str(i) + "_" + str(p) + ".png"), patU.reshape(patchSize,patchSize))
#
#     stop = timeit.default_timer()
#     fpTime = stop - start
#     print "Full program runtime:", fpTime
#
#
#
# # ##############################################################################
# #
# #     print "\n", "Pruning D1 KMEANS"
# #
# #     # store indicies of D1 clusters which have greater than 4 members
# #     pruned_gIndD1 = []
# #
# #     # store length of D1 clusters which have greater than 4 members
# #     pruned_gLenD1 = []
# #
# #     # store D1 clusters which have greater than 4 members
# #     # pruned_gClusD1 = np.array([])
# #     # pruned_gClusD1.shape=(0,cc)
# #     pruned_gClusD1 = {}
# #
# #     count = 0
# #     # store the indicies of the clusters with more than 3 patches
# #     for i in range(0,num_clustCenters):
# #         temp = stackPatchesD1[label.ravel() == i]
# #         [rt,ct] = temp.shape
# #         # print rt, ct
# #
# #         if(rt > 4):
# #             pruned_gIndD1.append(i)
# #             pruned_gLenD1.append(rt)
# #             # pruned_gClusD1 = np.vstack( (pruned_gClusD1,temp) )
# #             pruned_gClusD1[count] = temp
# #             count += 1
# #
# #     # print "\n", "num_clusters", num_clustCenters, "len pruned_gIndD1", len(pruned_gIndD1)
# #     prunedClusterCountD1 = (len(pruned_gIndD1))
# #     # print len(pruned_gIndD1)
# #     # print (pruned_gLenD1[0])
# #     # print pruned_gClusD1[0].shape[0]
# #     print "\n", "Removed cluster percentage for D1: ", float((num_clustCenters - prunedClusterCountD1)/num_clustCenters)
# #
# #
# #
# # ##############################################################################
# #
# #     print "\n", "Training on held-out N1 as all-negative examples", "\n"
# #
# #     # generate negative binary labels for all the N1 patches
# #     nLabelN1 = [0] * rn1
# #
# #     clfStore = {}
# #
# #     for i in range(0,1):
# #
# #         # generate positive binary labels for the patches in the current D1 cluster
# #         pLabelD1 = [1] * pruned_gLenD1[i]
# #
# #         # add the values from the two labels together
# #         fullLabel = pLabelD1 + nLabelN1
# #
# #         # extract the positive D1 patches in the current cluster
# #         d1patches = pruned_gClusD1[i]
# #         [rr,cc] = d1patches.shape
# #         print "d1: ", rr, pruned_gLenD1[i], "n1: ", rn1, "d1 + n1: ", rr + rn1, "labLen: ", len(fullLabel)
# #
# #         # vertically stack the positive cluster with D1 patches, over the negative cluster with N1 patches
# #         feats = np.vstack((d1patches,stackPatchesN1))
# #
# #         start_time = timeit.default_timer()
# #
# #         # train the linear SVM classifier on the pos_D1_patches and neg_N1_patches
# #         clf = svm.SVC(kernel='linear', probability=True)
# #         clf.fit(feats, fullLabel)
# #
# #         result = clf.predict(stackPatchesD2)
# #
# #         # print training time for ith classifier
# #         elapsed = timeit.default_timer() - start_time
# #
# #         print "Elapsed time for training classifier on cluster ", i, " :", elapsed
# #
# #         ind = []
# #         cnt = 0
# #         for i, el in enumerate(result):
# #             if(el != 0 and cnt < 5):
# #                 ind.append(i)
# #                 cnt += 1
# #
# #         newPatchClus = stackPatchesD2[ind,:]
# #
# #         print newPatchClus.shape
# #
# #         # j2 =  [r for r in result != 0]
# #         #
# #         # outval = np.extract(j2, result)
# #         #
# #         # print "vals not zero: ", outval
# #         # print "len outVal: ", len(outval)
# #         #
# #         # correct = np.count_nonzero(result)
# #         #
# #         # print "number of positives: ", correct
# #
# #         # clear the label in the end
# #         fullLabel = []
#
#
#
#
# ##############################################################################
#
#
#
#
# ##############################################################################
#     cv2.waitKey()
#
#     cv2.destroyAllWindows()