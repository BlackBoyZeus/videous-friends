class PegasusPoseUtils:
  def drawLandmarksOnImage(self, imageInput, poseProcessingInput):
    # Draw pose landmarks.
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    annotated_image = imageInput.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        poseProcessingInput.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    return annotated_image

  def doLandmarkConditionals(self, poseProcessingInput, activity2detect="Human"):
    '''
    IMPORTANT: The top is the 0 of the y-coordinate, so values increase as you go down in the image, aka the y-axis is inverted
    This means we invert the '<' and '>' when dealing with the y-axis 
    The left side is the 0 for the x-coordinate
    Currently we allow activity2detect to be either a string or list of a desired activity/ies
    e.g. activity2detect = "Soccer" or activity2detect = ["Basketball", "Soccer"]
    and we call the relative if statements by doing:  if 'Soccer' in activity2detect 
    this lets us count both string "Soccer" and a list of strings containing "Soccer", as True statements
    if you want to have multiple labels you have to add in functionality
    perhaps for multi-activity detection build a Truth array of sports [SoccerBoolean = True, BasketballBoolean = False] that triggers if any True 
    '''
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    if not poseProcessingInput.pose_landmarks: #if no human found it returns false, guard clause to prevent it failing trying to find landmarks
      return False

    print(activity2detect)
    if 'Human' in activity2detect:
      print("============ Human Detected ==================")
      return True #ie it passed the guard clause so it detected a pose, so it probably contains a human

    #if activity2detect == 'Soccer':
    if 'Soccer' in activity2detect:
      print("============ Soccer Detected ==================")
      (RIGHT_ANKLExvalue, RIGHT_ANKLEyvalue) = ( 
        poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x , 
        poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y 
        )
      (LEFT_ANKLExvalue, LEFT_ANKLEyvalue) = ( 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x , 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y 
          )
      (RIGHT_KNEExvalue, RIGHT_KNEEyvalue) = ( 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x , 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y 
          )
      (LEFT_KNEExvalue, LEFT_KNEEyvalue) = ( 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x , 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y 
          )
      if (RIGHT_ANKLEyvalue < RIGHT_KNEEyvalue) or (RIGHT_ANKLEyvalue < LEFT_KNEEyvalue) or (LEFT_ANKLEyvalue < RIGHT_KNEEyvalue) or (LEFT_ANKLEyvalue < LEFT_KNEEyvalue): #y-axis inverted so less than means if ankles are above
        return True
      else:
        return False

    #if activity2detect == 'Basketball':
    if 'Basketball' in activity2detect:
      print("============ Basketball Detected ==================")
      (NOSExvalue, NOSEyvalue) = ( 
        poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x , 
        poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
        )
      (RIGHT_WRISTxvalue, RIGHT_WRISTyvalue) = ( 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x , 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y 
          )
      (LEFT_WRISTxvalue, LEFT_WRISTyvalue) = ( 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x , 
          poseProcessingInput.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y 
          )
      #print("RIGHT_WRISTyvalue=", RIGHT_WRISTyvalue, "  LEFT_WRISTyvalue= ", LEFT_WRISTyvalue, "  NOSEyvalue= ", NOSEyvalue)
      if (RIGHT_WRISTyvalue < NOSEyvalue) or (LEFT_WRISTyvalue < NOSEyvalue): #y-axis inverted so less than means if wrists are above
        return True
      else:
        return False

class PegasusVideoUtils:
  def neighborhoodModification(self, labelListInput, radius2flip, fps=30):
    '''flip neighborhood -ie radius in frameLabelList- to 1's
    Done to prevent choppiness
    radius2flip = number of frames to flip
    Deffaults to 30 fps. radius2flip/FPS = number of seconds to add
    '''

    frameLabelListCopy = [0]*len(labelListInput) #start it out as empty (aka devoid of detection instances)
    
    for ii in range(len(labelListInput)):
      if ii < radius2flip:
        if 1 in labelListInput[ii : ii + radius2flip + 1]: 
          #ie if one of the next radius2flip frames contain a 1 (positive detection) then its ok to go ahead and change current to a 1, otherwise leave it 0
          frameLabelListCopy[ii] = 1
        #else:
        #  frameLabelListCopy[ii] = 0
      else:
        if 1 in labelListInput[ii - radius2flip : ii + radius2flip + 1]:
          #basically the past frame case 
          frameLabelListCopy[ii] = 1
        
    return frameLabelListCopy

class PegasusImageUtils:
  def resizeTARGET(self, image, TARGET_HEIGHT=256, TARGET_WIDTH=256):
    import cv2
    import math 
    h, w = image.shape[:2]
    if h < w:
      img = cv2.resize(image, (TARGET_WIDTH, math.floor(h/(w/TARGET_WIDTH))))
    else:
      img = cv2.resize(image, (math.floor(w/(h/TARGET_HEIGHT)), TARGET_HEIGHT))
    return img

class PegasusMain:
  def __init__(self):
    print("Ran Pegasus")

  def analyzeVideo(self, videoPath, sportSelection="Basketball", staticParam=False, minDetectionConfidenceParam=0.5, modelComplexityParam=0, TARGET_HEIGHT=256, TARGET_WIDTH=256, drawLandmarksToVideo=False, breakAtCountLimit=False, skipFrameInterval= 1):
    import cv2
    import math
    import numpy as np
    
    pegasusPoseUtils = PegasusPoseUtils()
    pegasusImageUtils = PegasusImageUtils()
    
    import time
    start = time.time()
    
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    capture = cv2.VideoCapture(videoPath)
    
    frameNr = 0
    frameNrActivated = 0 #counter for when we excise frames. Still want to label them 1,2,3,... 

    #TARGET_HEIGHT = 256
    #TARGET_WIDTH = 256
    #drawLandmarksToVideo = False
    #breakAtCountLimit = False
    #we skip the algo for frameNr % skipFrameInterval != 0 so if skipFrameInterval = 5 we skip 4 of 5 frames 
    #set to 1 to collect all frames (aka default)
    #set to 2 to collect half
    #set to 3 to collect one-third, etc.... 
    #skipFrameInterval = 1
    #staticParam = False
    #minDetectionConfidenceParam = 0.5
    #modelComplexityParam = 0 #0,1,2 0 lightest fastest 2 heaviest slowest

    frameLabelList = []

    #Docs: https://google.github.io/mediapipe/solutions/pose.html#static_image_mode
    #with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    with mp_pose.Pose(static_image_mode=staticParam, min_detection_confidence=minDetectionConfidenceParam, model_complexity=modelComplexityParam) as pose:
        while (True):
            success, frame = capture.read()
            if not success:
                break

            frameNr += 1
            if frameNr % skipFrameInterval != 0:
              frameLabelList.append(0) #still want to build this list so we add a 0 
              continue #aka skip this frame

            image = pegasusImageUtils.resizeTARGET(frame, TARGET_HEIGHT, TARGET_WIDTH)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if breakAtCountLimit == True and frameNr == 2001: #break limit for tests, ends it at frame 100 since 101 won't be saved 
              break

            if drawLandmarksToVideo == True:
              annotated_image = pegasusPoseUtils.drawLandmarksOnImage(imageInput=image, poseProcessingInput=results)
              #cv2_imshow(annotated_image)
              #cv2.imwrite(f"outputFrames/frame_{frameNr}.png", annotated_image)
              continue #use continue so it doesn't run anything below if this conditional is activated
            
            if pegasusPoseUtils.doLandmarkConditionals(poseProcessingInput=results, activity2detect=sportSelection):
              #save the frame if the sports pose conditional is satisfied
              frameNrActivated += 1
              #cv2.imwrite(f"outputFrames/frame_{frameNrActivated}.png", frame)
              print(f"Activity detected and frame saved for frame number {frameNr}, saved as activation counter number {frameNrActivated}")

              frameLabelList.append(1)
            else:
              print("Requested activity not detected")
              frameLabelList.append(0)

            #Create copies of all original frames for later reference when building a new vid with frameLabelList
            #cv2.imwrite(f"originalFrames/frame_{frameNr}.png", frame)
            #print(frameLabelList)
            print(f"Processed frame number {frameNr}")

    capture.release()

    end = time.time()
    print(f"Time (seconds) to process for {frameNr -1} frames: ", end - start, f"  for parameter set TARGET_HEIGHT = {TARGET_HEIGHT} TARGET_WIDTH = {TARGET_WIDTH} drawLandmarksToVideo = {drawLandmarksToVideo} breakAtCountLimit = {breakAtCountLimit} skipFrameInterval = {skipFrameInterval} staticParam = {staticParam} minDetectionConfidenceParam = {minDetectionConfidenceParam} modelComplexityParam = {modelComplexityParam}")

    return frameLabelList

  def buildHighlightVideo(self, videoPath, frameLabelList, outVideoPath="edittedVideo.mp4", numSecondsPadding = 1, skipFrameInterval=1):
    pegasusVideoUtils = PegasusVideoUtils()

    import cv2
    capture = cv2.VideoCapture(videoPath)
    fpsRate = int( capture.get(cv2.CAP_PROP_FPS) )

    print("FPS of input video: ", fpsRate)
    print("Length of frameLabelList: ", len(frameLabelList) )

    referenceLabelList = pegasusVideoUtils.neighborhoodModification(frameLabelList, numSecondsPadding*int(fpsRate), fps=fpsRate)
    print(len(referenceLabelList))

    frameNr = 0
    img_array = [] #array of frames of the new video

    while (True):
      success, frame = capture.read()
      if not success:
          break
      
      frameNr += 1

      #if frameNr % skipFrameInterval != 0:
      #  continue

      try:
        #print(frameNr - 1)
        if referenceLabelList[frameNr - 1] == 1: #aka only add them to the video if their label indicates a detection 
          height, width, layers = frame.shape
          size = (width,height)

          img_array.append(frame)
      except:
        #in the case where breakAtCountLimit=True ie we only analyze a subsection of the video, the number of frames in capture can exceed len( referenceLabelList ) 
        #so we end the loop and build the video
        #plus any number of things can go wrong in video processing so leaving in this try/except might be prudent
        break 

    try:
      print("Building video....")
      out = cv2.VideoWriter(outVideoPath,cv2.VideoWriter_fourcc(*'DIVX'), fps=fpsRate, frameSize=size)
      
      for i in range(len(img_array)):
          out.write(img_array[i])
      out.release()
    except:
      print("No matching frames found")

  def runMain(self, videoFilePath, activitySelection, complexityReducer=1, numSecondsPaddingToApply=1, outputVideoPath="editedVideo.mp4"):
    frameLabelList0 = self.analyzeVideo(
        videoPath= videoFilePath, 
        sportSelection= activitySelection, 
        staticParam=False, 
        minDetectionConfidenceParam=0.5, 
        modelComplexityParam=0, 
        TARGET_HEIGHT=256, 
        TARGET_WIDTH=256, 
        drawLandmarksToVideo=False, 
        breakAtCountLimit=False, 
        skipFrameInterval= complexityReducer #higher = less frames = less complex
        )
    
    self.buildHighlightVideo(
        videoPath= videoFilePath, 
        frameLabelList= frameLabelList0, 
        outVideoPath=outputVideoPath, 
        numSecondsPadding = numSecondsPaddingToApply,
        skipFrameInterval = complexityReducer
        )