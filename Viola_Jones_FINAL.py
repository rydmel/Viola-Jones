'''
Ryan D'Mello
CMSC 254
'''


import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
import time

######################### BEGINNING of training code #######################################################

global images                    # array of 4000 images, each a 64 x 64 array
global labels                    # array of 4000 labels 1 (face) and -1 (background)
global integral_images           # array of 4000 64 x 64 image integrals
global features                  # a list of features
global FaceBkgrd_Label_tuple     # tuple of [images, labels]
global StrongLearner             # Final AdaBoost-ed weak learner set
global THETA                     # The Big Theta used as "correction" in calculating h(x) to articfially boost False Negative Rate
global Alphas                    # the alphas for the final weak learner set (generated from AdaBoost)


# Loads an image file into a 64 x 64 grayscale array
def load_file(filename):
   temp_image = Image.open(filename).convert('L')  #Converts to grayscale
   return np.array(temp_image)

# Uses load_file to read ALL images in a directory to respective arrays
def get_images(directory):
    temp_array = [load_file(directory + '/'+ filename) for filename in os.listdir(directory)]
    return(temp_array)

# Assigns +1 and -1 to the image and background files, respectively
def get_labels(directory):
    if (directory == 'faces'):
        temp_array = [1 for filename in os.listdir(directory)]
    else:
        temp_array = [-1 for filename in os.listdir(directory)]
    return(temp_array)
    
# Creates a tuple of two arrays: 4000-image array and 4000-label array
def load_data(faces_dir, background_dir):
    global images
    global labels
    
    faces = get_images(faces_dir)
    background = get_images(background_dir)
    face_labels = get_labels(faces_dir)
    background_labels = get_labels(background_dir)
    images = np.concatenate((faces, background),axis=0)
    labels = np.concatenate((face_labels, background_labels),axis=0)
    return(images,labels)
    
# Creates an array consisting of the "integrals" of each of the 4000 images
def compute_integral_image(imgs):
   l = len(imgs)
   newImgs = [np.cumsum(np.cumsum(imgs[i], axis=0), axis=1) for i in range(l)]
   finalArr = np.array(newImgs).astype(int)
   return finalArr

# Generate a feature list, each feature being p x q. Here, n = 64 but a
# general n was chosen for testing purposes. Returns coordinate list of ULs
# (Upper Left shaded), ULw (ULwhite), LRs, and LRw. Shaded p x q rectangle
# is above white p x q (so full size of shaded + unshaded is actually p x 2q)
# The stride is essentially the separation between consecutive features.
# max_features is the maximum number of features to be generated
def feature_list(n, p, q, stride,max_features):
    feat_list = []
    for x in range(p,n,p + stride):
        for y in range(q,n,q + stride):
            for l in range(0,n,q + 1):
                for h in range(0,n,p + 1):
                    if( x + h >= n or 2 * y + l >= n):
                        break
                    ULs  = (l, h)
                    ULw = (l + y, h)
                    LRw= (l + y*2, h + x)
                    LRs = (l + y, h + x)
                    feat_list.append([ULs,LRs,ULw,LRw])
                    if (len(feat_list) >= max_features):
                        return feat_list
    return feat_list

# Use the integral images to compute the value of the feature in feat_lst
# whose index is feat_idx on the image in images[] whose index is img_idx
# The first parameter int_img_rep should be passed as integral_images
def compute_feature_on_image(int_img_rep, feat_lst, feat_idx, img_idx):
    image = int_img_rep[img_idx]
    ULs, LRs, ULw, LRw = feat_lst[feat_idx]
    UpperRect = image[LRs[0]][LRs[1]] + image[ULs[0]][ULs[1]] - image[ULs[0]][LRs[1]] - image[LRs[0]][ULs[1]]
    LowerRect = image[LRw[0]][LRw[1]] + image[ULw[0]][ULw[1]] - image[ULw[0]][LRw[1]] - image[LRw[0]][ULw[1]]
    return UpperRect - LowerRect

# Now use compute_feature_on_image to return an 4000 x 1 numpy matrix
# whose elements are the feature evaluations on each of the 4000 images
def compute_feature(int_img_rep, feat_lst, feat_idx):
    rng = len(int_img_rep)
    return [compute_feature_on_image(int_img_rep, feat_lst, feat_idx, img_idx) for img_idx in range(rng)]
    
# Compute the optimal p and theta values for a given feature index using
# the method sketched out on slide 27 in the Boosting slides
def opt_p_theta(int_img_rep, feat_lst, weights, feat_idx): 
    global labels
    feature_evals = compute_feature(int_img_rep,feat_lst,feat_idx)
    N = len(feature_evals)                                      # N = 4000 for this assignment
    index_sortevals = np.argsort(feature_evals,axis=0)          # yields INDICESthat would sort feature_evals
    feature_evals = np.array(feature_evals)[index_sortevals]    # this sorts feature_evals
    corresponding_weights = np.array(weights)[index_sortevals]  # rearrange weights to match sorting
    corresponding_labels = np.array(labels)[index_sortevals]    # rearrange labels to match sorting
    p_val = np.empty(shape=N)                                   # create uninitialized p value array of size N
    eps_val = np.empty(shape=N)                                 # create uninitialized epsilon array of size N
    
    # Now loop through the N images and calculate the S+, T+, S-, and T- as in slide #28 
    for i in range(N):
        plus = np.empty(shape=N)
        for j in range(N):
            if (corresponding_labels[j] >= 0):
                plus[j] = 1
            else:
                plus[j] = 0
        minus = [1 - t for t in plus]
        Splus = np.sum([plus[q] * corresponding_weights[q] for q in range(i + 1)])
        Tplus = np.sum([plus[q] * corresponding_weights[q] for q in range(N)])
        Sminus = np.sum([minus[q] * corresponding_weights[q] for q in range(i + 1)])
        Tminus = np.sum([minus[q] * corresponding_weights[q] for q in range(N)])
        first_arg = Splus + Tminus - Sminus
        second_arg = Sminus + Tplus - Splus
        if (first_arg < second_arg):
            eps_val[i] = first_arg
            p_val[i] = 1
        else:
            eps_val[i] = second_arg
            p_val[i] = -1
    mineps_index = np.argmin(eps_val)
    if (mineps_index == (N - 1)):
        return(feature_evals[mineps_index],p_val[mineps_index])
    else:
        return(0.5*(feature_evals[mineps_index] + feature_evals[mineps_index + 1]),p_val[mineps_index])
        
# Compute the predictions of a given weak learner
def eval_learner(int_img_rep, feat_lst, feat_idx, p, theta): 
    N = len(int_img_rep) 
    predictions = np.empty(shape=N)                 # create uninitialized predictions array of size N
    #Now loop through and fill in the predictions array (use formula on slide #27 in Boosting slides)
    for i in range(N):
        raw_prediction = p * (compute_feature_on_image(int_img_rep, feat_lst, feat_idx, i) - theta)
        if (raw_prediction >= 0):
            predictions[i] = 1
        else:
            predictions[i] = -1
    return predictions

# Compute the weighted error rate of a given weak learner
def error_rate(int_img_rep, feat_lst, weights, feat_idx, p, theta):
    global labels
    N = len(int_img_rep)
    # First call eval_learner to get the predictions
    predictions = eval_learner(int_img_rep, feat_lst, feat_idx, p, theta)
    weighted_error = 0
    for i in range(N):
        if (predictions[i] != labels[i]):
            factor = 1
        else:
            factor = 0
        weighted_error += weights[i] * factor
    return weighted_error  

# Find and return i, p, and theta values of the optimal weak learner,
# where "optimal" means lowest error rate
def opt_weaklearner(int_img_rep, weights, feat_lst):   
    global features
    F = len(features)
    error_arr = np.empty(shape=F)
    for i in range(F):
        if (i % 5 == 0):
            print('         opt_weaklearner: starting feature #',i + 1, "of",F)
        theta,p = opt_p_theta(int_img_rep, feat_lst, weights, i)
        error_arr[i] = error_rate(int_img_rep, feat_lst, weights, i, p, theta)
    opt = np.argmin(error_arr)                # Index with least error
    theta,p = opt_p_theta(int_img_rep, feat_lst, weights, opt)
    return(opt,p,theta)
    
# Update the weights using the formulas in Part 6 of homework
def update_weights(weights, error_rate, y_pred, y_true): 
    W = len(weights)
    updated_weights = np.empty(shape=W)
    for i in range(W):
        if (error_rate < 0.05):
            Z_t = 1
            Alpha_t = 0
        else:
            Z_t = 2 * np.sqrt(error_rate*(1 - error_rate))
            Alpha_t = 0.5 * np.log((1 - error_rate)/error_rate)
        update_factor = np.exp(-1 * Alpha_t * y_true[i] * y_pred[i])
        updated_weights[i] = (update_factor * weights[i])/Z_t
    return updated_weights   

# Use the weak learners to compute the "strong learner" prediction
def strongPrediction(weakLearners,alphas, image_idx, int_img_rep, feat_lst):
    A = len(alphas)         # number of weak learners
    strong_pred = 0
    for i in range(A):
        feat_idx = int(weakLearners[i][0])
        p = int(weakLearners[i][1])
        theta = weakLearners[i][2] 
        H = compute_feature_on_image(int_img_rep, feat_lst, feat_idx, image_idx)
        value = p * (H - theta)
        strong_pred += value * alphas[i]
    return strong_pred
        
# Compute the False Positive Rate of a set containing weaklearners.
# Each weaklearner is characterized by a feature index and its theta and 
# p values
def FPRcompute(weaklearners,alphas, y_true, int_img_rep, feat_lst):
    N = len(int_img_rep)
    FPcount = 0
    Neg_count = 0
    for i in range(N):
        if (y_true[i] == -1):
            Neg_count += 1          # final value of neg_count should be 2000 (backgrounds)
            prediction = strongPrediction(weaklearners,alphas,i,int_img_rep, feat_lst)
            if (prediction >= 0):
                FPcount += 1
    return(FPcount/Neg_count)
    
# Compute the False Negative Rate of a set containing weaklearners.
def FNRcompute(weaklearners,alphas, y_true, int_img_rep, feat_lst):
    N = len(int_img_rep)
    FNcount = 0
    Pos_count = 0
    for i in range(N):
        if (y_true[i] == 1):
            Pos_count += 1          # final value of pos_count should be 2000 (faces)
            prediction = strongPrediction(weaklearners,alphas,i,int_img_rep, feat_lst)
            if (prediction < 0):
                FNcount += 1
    return(FNcount/Pos_count)
    
# We now have all the helper functions to perform AdaBoost
def AdaBoost(int_img_rep, y_true, feat_lst):
    W = 10           # Max number of weak learners
    N = len(int_img_rep)
    weights = [1/N for i in range(N)]           # initial equal weights adding up to 1
    weakLearners = np.empty(shape=(W,3))        # placeholders for W weak learners
    alphas = np.empty(shape=W)   
    FPrate = 1                                  # false positive rate
    FNrate = 1                                  # false negative rate
    iterations = 0
    FParray = []                                # store false positive rates for plotting later
    FNarray = []  
                              # store false negative rates for plotting later
    print("START performing AdaBoost. Iterations TAKE LONG!! Alert is provided on entering every FIFTH feature.")
    while ((((FPrate > 0.2) or (FNrate > 0.1)) and iterations < W) or iterations < 4):
        print("Starting iteration#",iterations + 1,"in AdaBoost - calling opt_weaklearner .....")
        feat_idx,p,theta = opt_weaklearner(int_img_rep,weights,feat_lst)
        predictions = eval_learner(int_img_rep, feat_lst, int(feat_idx) ,int(p), theta)
        weighted_error = error_rate(int_img_rep, feat_lst, weights, int(feat_idx) ,int(p), theta)
        alpha = .5 * np.log(((1 - weighted_error)/weighted_error))
        weights = update_weights(weights,weighted_error,predictions,y_true)
        alphas[iterations] = alpha
        print("Optimal learner index:",feat_idx,"   Adding it to weaklearner list - p and theta are (respectively)",p,theta)
        weakLearners[iterations][0] = feat_idx
        weakLearners[iterations][1] = p
        weakLearners[iterations][2] = theta
        FPrate = FPRcompute(weakLearners[:iterations + 1],alphas[:iterations + 1], y_true, int_img_rep, feat_lst)
        FParray.append(FPrate)
        FNrate = FNRcompute(weakLearners[:iterations + 1],alphas[:iterations + 1], y_true, int_img_rep, feat_lst)
        FNarray.append(FNrate)
        print("FALSE NEGATIVE/FALSE POSITIVE rates after iteration #", iterations + 1, "are:",FNrate,"/", FPrate,"      Weights have been updated\n")
        #print("Weights have been updated. First 10 new weights are:")          # for debugging
        #print(weights[0],weights[1],weights[2],weights[3],weights[4],weights[5],weights[6],weights[7],weights[8],weights[9],"\n")  # for debugging
        iterations += 1
        
    # Plot the false negative and false positive rates
    #
    print ("Plots for FALSE NEGATIVES (BLUE) and FALSE POSITIVES (ORANGE) are shown below (x-axis = iteration #):")
    plt.plot(FNarray)
    plt.plot(FParray)
    
    # Get the minimum strong prediction  
    minval = 100000000      # very large number
    for i in range(N):
        if(y_true[i] == 1):     # This is a face
            strongval = strongPrediction(weakLearners[:iterations],alphas[:iterations], i,int_img_rep,feat_lst)
            if(strongval < minval):
                minval = strongval
    return (weakLearners[:iterations], alphas[:iterations],minval)
 
# The post-AdaBoost THETA adjusted "strong learner" prediction
def FinalStrongPrediction(StrongLearner,alphas, image_idx, int_img_rep, feat_lst):
    A = len(alphas)         # number of weak learners
    strong_pred = 0
    global THETA
    for i in range(A):
        feat_idx = int(StrongLearner[i][0])
        p = int(StrongLearner[i][1])
        littletheta = StrongLearner[i][2]
        H = compute_feature_on_image(int_img_rep, feat_lst, feat_idx, image_idx)
        value = p * (H - littletheta)
        strong_pred += value * alphas[i]
    return strong_pred - (THETA/1.6)     # Experiment with denominators to reach FNR near zero and FPR < 30%

# THETA adjusted Final FPR computation
def FinalFPRcompute(StrongLearner,alphas, y_true, int_img_rep, feat_lst):
    N = len(int_img_rep)
    FPcount = 0
    Neg_count = 0
    for i in range(N):
        if (y_true[i] == -1):
            Neg_count += 1          # final value of neg_count should be 2000 (backgrounds)
            prediction = FinalStrongPrediction(StrongLearner,alphas,i,int_img_rep, feat_lst)
            if (prediction >= 0):
                FPcount += 1
    return(FPcount/Neg_count)

# THETA adjusted Final FNR computation
def FinalFNRcompute(StrongLearner,alphas, y_true, int_img_rep, feat_lst):
    N = len(int_img_rep)
    FNcount = 0
    Pos_count = 0
    for i in range(N):
        if (y_true[i] == 1):
            Pos_count += 1          # final value of pos_count should be 2000 (faces)
            prediction = FinalStrongPrediction(StrongLearner,alphas,i,int_img_rep, feat_lst)
            if (prediction < 0):
                FNcount += 1
    return(FNcount/Pos_count)
    
######################### END of "training" code #######################################################
    
   
    
    
######################### BEGINNING of testing code ####################################################
    
global Slider                            # this is a 64 x 64 grayscale frame that "slides" up and down the image
global TLslider                          # coordinates of top left corner of slider, starts at (0,0)
global GSarray                           # grayscale array of test image
global RGBarray                          # RGB array of test image
global INTEGarray                        # integral image based on slider
global temprgb                           # modified picture showing colored squares
Slider = np.empty(shape=(64,64),dtype=int)
INTEGarray = np.empty(shape=(64,64),dtype=int)


# Store a grayscale AND RGB version of the test image. The grayscale version
# is used for the face detection, and the RGB is used to draw colored boxes
def load_image(filename):
    global GSarray                   
    global RGBarray
    GSimage = Image.open(filename)        # Gray Scale test image
    RGBimage = GSimage.convert('RGB')     # RGB version, so we can draw colored boxes
    GSarray = np.array(GSimage)
    RGBarray = np.array(RGBimage)
        

# Using the coordinates of its top left corner, fill the slider so that
# it captures a 64 x 64 slice of the grayscale picture
def fill_slider(TLcorner):
    global Slider
    global INTEGarray
    row_val = TLcorner[0]
    col_val = TLcorner[1]
    if ((row_val + 64 > 1280) or (col_val + 64 > 1600)):
        return -1               # can't fill this slider because it extends beyond the picture dimensions
    for rowoffset in range(64):
        for coloffset in range(64):
            row = row_val + rowoffset
            col = col_val + coloffset
            Slider[rowoffset][coloffset] = GSarray[row][col]
            INTEGarray = [np.cumsum(np.cumsum(Slider, axis=0), axis=1)]
            
    return 1                    # slider could be filled
    
# Draw a square of length "length" pixels on the RGB image (via the RGBarray) whose 
# top left corner is TLcorner and whose color is an rgb triad. Thickness of line 
# is specified in pixels. Checking that the box does not extend beyond picture 
# dimensions is done BEFORE calling this function
def drawColorSquare(TLcorner,thickness,length, rgb):
    global temprgb
    row_val = TLcorner[0]
    col_val = TLcorner[1]
    temprgb = RGBarray
    # Draw left and right lines of square
    for rowoffset in range(length):
        for coloffset in range(thickness): 
            row = row_val + rowoffset
            left_col = col_val + coloffset
            right_col = col_val + length - 1 - coloffset
            temprgb[row][left_col] = rgb                   # left line
            temprgb[row][right_col] = rgb                  # right line
    # Draw top and bottom lines of square
    for coloffset in range(length):
        for rowoffset in range(thickness): 
            col = col_val + coloffset
            top_row = row_val + rowoffset
            bottom_row = row_val + length - 1 - rowoffset
            temprgb[top_row][col] = rgb                     # top line
            temprgb[bottom_row][col] = rgb                  # bottom line
    #plt.imshow(temprgb)                       # used for testing
    
# compute the value of a feature on the slider
def compute_feature_on_slider(image, feat_lst, feat_idx):
    ULs, LRs, ULw, LRw = feat_lst[feat_idx]
    #print(ULs,LRs,ULw,LRw)                          # used for testing
    UpperRect = image[0][LRs[0]][LRs[1]] + image[0][ULs[0]][ULs[1]] - image[0][ULs[0]][LRs[1]] - image[0][LRs[0]][ULs[1]]
    LowerRect = image[0][LRw[0]][LRw[1]] + image[0][ULw[0]][ULw[1]] - image[0][ULw[0]][LRw[1]] - image[0][LRw[0]][ULw[1]]
    return UpperRect - LowerRect

# get the prediction on the slider: is it a face or not?
def SliderPrediction(StrongLearner,alphas, feat_lst):
    A = len(alphas)         # number of weak learners
    strong_pred = 0
    global THETA
    for i in range(A):
        feat_idx = int(StrongLearner[i][0])
        p = int(StrongLearner[i][1])
        littletheta = StrongLearner[i][2]
        H = compute_feature_on_slider(INTEGarray, features, feat_idx)
        value = p * (H - littletheta)
        strong_pred += value * alphas[i]
    return strong_pred - (THETA/1.6)  
    
# This function runs the 64 x 64 slider through the entire image, checking at
# each step whether it contains a face. row_disp and col_disp are the horizontal
# and vertical displacements for the movement of the slider from one position 
# to its next position
def runSlider(row_disp,col_disp):
    global Slider
    slidecount = 0                      # tracks how many times the slider slides
    facecount = 0
    tempcount = 0
    print("Faces were detected in 64 x 64 pixel squares whose TOP LEFT CORNERS have the following coordinates:")
    for row_val in range(192,640,row_disp):
        if (row_val + 64 > 1280):
            break
        for col_val in range(192,1472,col_disp):
            if (col_val + 64 > 1600):
                break
            tempcount += 1
            if (fill_slider([row_val,col_val]) == 1):
                slidecount += 1
                if (SliderPrediction(StrongLearner,Alphas, features) >= 0):
                    print("[",row_val,",",col_val,"]")
                    drawColorSquare([row_val,col_val],10,60, [255,69,0])
                    facecount += 1
    plt.imshow(temprgb)
    print("\nTotal number of faces detected by Viola-Jones:", facecount, "              Actual total:",50)
    print("\nHere is a picture of the faces bounded by orange squares\n")
    return(tempcount, slidecount, facecount)
    
######################### END of testing code ##########################################################
       
######################### Main program starts here #####################################################    
def main():
    global integral_images
    global FaceBkgrd_Label_tuple
    global features
    global labels
    global StrongLearner             # Final AdaBoost-ed weak learner set
    global THETA                     # The Big Theta used as "correction" in calculating h(x) to articfially boost False Negative Rate
    global Alphas                    # the alphas for the final weak learner set (generated from AdaBoost)
    max_features = 60
    
    # Sleep statements are to pause bursts of output, so viewer can keep up
    print("START loading faces and backgrounds and form label array (+1 and -1) ....")
    FaceBkgrd_Label_tuple = load_data('faces','background')
    print("FINISHED loading faces/backgrounds and labels into a tuple of arrays\n")
    time.sleep(2)
    
    print("START computing the integral array .....")
    integral_images = compute_integral_image(FaceBkgrd_Label_tuple[0])
    print("FINISHED computing the integral array\n")
    time.sleep(2)
     
    print("START forming the feature list .....")
    features = feature_list(64,10,6,2,max_features)   # coordinate list for 4 x 2 pairs of shaded/unshaded; stride 2
    print("FINISHED forming the feature list\n")
    time.sleep(2)
    
    ############# START the TRAINING part of the program ############################
    StrongLearner, Alphas, THETA = AdaBoost(integral_images, labels, features)
    FINAL_FNR = FinalFNRcompute(StrongLearner,Alphas, labels, integral_images, features)
    FINAL_FPR = FinalFPRcompute(StrongLearner,Alphas, labels, integral_images, features)
    print("AdaBoost HAS COMPLETED!\n")
    print("THETA-adjusted FALSE NEGATIVE rate is:", FINAL_FNR,"    THETA-adjusted FALSE POSITIVE rate is:", FINAL_FPR)
    ############# END the TRAINING part of the program ##############################
    
    
    ############# START the TESTING part of the program (on test_img image) ###########
    load_image("test_img.jpg")
    
    # Run a 64x 64 slider up and down the test image and use the strong learners identified
    # in the training part above to detect faces. Print out the coordinates of the upper
    # left corner of the 64 x 64 frame in which a face is detected, and also draw an
    # orange-colored box outlining each such frame. Print number of faces detected and
    # actual number of faces                 
    
    test_image_result = runSlider(64,64)
    ############# END the TESTING part of the program (on test_img image) #############

main()