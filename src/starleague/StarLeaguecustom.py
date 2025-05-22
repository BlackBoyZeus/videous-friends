import tkinter
from tkinter import filedialog
import customtkinter
import os
#import mediapipe as mp
import cv2
import math
import numpy as np
import time
import requests
import json
import datetime

def tk_write(tk_string1): #Makes a popup displaying text string tk_string1. Effectively acts as an alert, when migrating to web or mobile oriented UI you can change this to a js or other alert
    window = tkinter.Tk()
    window.title("StarLeague")
    window.geometry('800x400')
    tk_string1 = str(tk_string1)
    text1 = tkinter.Label(window, text= tk_string1)
    text1.grid(column=0, row=0) 
    window.mainloop()

# ======================================== PEGASUS ALGOS ========================================
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

    #print(activity2detect)
    if 'Human' in activity2detect:
      #print("============ Human Detected ==================")
      return True #ie it passed the guard clause so it detected a pose, so it probably contains a human

    #if activity2detect == 'Soccer':
    if 'Soccer' in activity2detect:
      #print("============ Soccer Detected ==================")
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
      #print("============ Basketball Detected ==================")
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
      ##print("RIGHT_WRISTyvalue=", RIGHT_WRISTyvalue, "  LEFT_WRISTyvalue= ", LEFT_WRISTyvalue, "  NOSEyvalue= ", NOSEyvalue)
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
    #import cv2
    #import math 
    h, w = image.shape[:2]
    if h < w:
      img = cv2.resize(image, (TARGET_WIDTH, math.floor(h/(w/TARGET_WIDTH))))
    else:
      img = cv2.resize(image, (math.floor(w/(h/TARGET_HEIGHT)), TARGET_HEIGHT))
    return img

class PegasusMain:
  def __init__(self):
    pass

  def analyzeVideo(self, videoPath, sportSelection="Basketball", staticParam=False, minDetectionConfidenceParam=0.5, modelComplexityParam=0, TARGET_HEIGHT=256, TARGET_WIDTH=256, drawLandmarksToVideo=False, breakAtCountLimit=False, skipFrameInterval= 1):
    #import cv2
    #import math
    #import numpy as np
    
    pegasusPoseUtils = PegasusPoseUtils()
    pegasusImageUtils = PegasusImageUtils()
    
    #import time
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
              #print(f"Activity detected and frame saved for frame number {frameNr}, saved as activation counter number {frameNrActivated}")

              frameLabelList.append(1)
            else:
              #print("Requested activity not detected")
              frameLabelList.append(0)

            #Create copies of all original frames for later reference when building a new vid with frameLabelList
            #cv2.imwrite(f"originalFrames/frame_{frameNr}.png", frame)
            ##print(frameLabelList)
            #print(f"Processed frame number {frameNr}")

    capture.release()

    end = time.time()
    #print(f"Time (seconds) to process for {frameNr -1} frames: ", end - start, f"  for parameter set TARGET_HEIGHT = {TARGET_HEIGHT} TARGET_WIDTH = {TARGET_WIDTH} drawLandmarksToVideo = {drawLandmarksToVideo} breakAtCountLimit = {breakAtCountLimit} skipFrameInterval = {skipFrameInterval} staticParam = {staticParam} minDetectionConfidenceParam = {minDetectionConfidenceParam} modelComplexityParam = {modelComplexityParam}")

    return frameLabelList

  def buildHighlightVideo(self, videoPath, frameLabelList, outVideoPath="edittedVideo.mp4", numSecondsPadding = 1, skipFrameInterval=1):
    pegasusVideoUtils = PegasusVideoUtils()

    #import cv2
    capture = cv2.VideoCapture(videoPath)
    fpsRate = int( capture.get(cv2.CAP_PROP_FPS) )

    #print("FPS of input video: ", fpsRate)
    #print("Length of frameLabelList: ", len(frameLabelList) )

    referenceLabelList = pegasusVideoUtils.neighborhoodModification(frameLabelList, numSecondsPadding*int(fpsRate), fps=fpsRate)
    #print(len(referenceLabelList))

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
        ##print(frameNr - 1)
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
      #print("Building video....")
      out = cv2.VideoWriter(outVideoPath,cv2.VideoWriter_fourcc(*'DIVX'), fps=fpsRate, frameSize=size)
      
      for i in range(len(img_array)):
          out.write(img_array[i])
      out.release()
    except:
      tk_write("No frames matching your desired activity were found. \nPlease adjust your settings and try again. ")
      #print("No matching frames found")

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

# ======================================== GUMROAD GATEWAY ========================================

def checkGumroadLicenseValidity(numberOfUsesAllowedPerYear=1000):
    #import requests
    #import json
    #import datetime

    #These 3 are custom to your app:
    sellerIDstring = "a3pTZLrSPvxOAmV2e7-7ow=="
    productIDstring = "gzNFpBwohWBTPdoN-HdkZg==" #So they can't different use key from some seller
    productPermalinkstring = "Star"

    gum_base_url = "https://api.gumroad.com/v2/licenses/verify"
    productPermalink = "Star"

    licenseKeyfile = open("licensekey.txt" , "r")
    licenseKey = licenseKeyfile.read()
    licenseKeyfile.close()
    licenseKey = str(licenseKey)

    productPermalink = str(productPermalink) 
    licenseKey = str(licenseKey).replace(" ", "").replace("\n", "")

    #API POST request URL:
    gum_url = gum_base_url + "?product_permalink=" + productPermalink + "&license_key=" + licenseKey 

    r = requests.post( url = gum_url) 

    gumjson = r.json()

    userCurrentYear = int( datetime.datetime.now().year )
    numUsesTimesYearsUsed = int(numberOfUsesAllowedPerYear)  * int(userCurrentYear - int( gumjson["purchase"]["created_at"][0:4] ) + 1)
    #print(numUsesTimesYearsUsed)

    #ACCESS CONTROL
    if ( 
        gumjson["success"] == True
        and gumjson["uses"] < numUsesTimesYearsUsed #int(numberOfUsesAllowedPerYear) 
        and gumjson["purchase"]["seller_id"] == sellerIDstring
        and gumjson["purchase"]["product_id"] == productIDstring
        and gumjson["purchase"]["permalink"] == productPermalinkstring
        and gumjson["purchase"]["refunded"] == False
        and gumjson["purchase"]["disputed"] == False
        and gumjson["purchase"]["dispute_won"] == False
        and gumjson["purchase"]["subscription_ended_at"] == None
        and gumjson["purchase"]["subscription_failed_at"] == None):

        return True
    else:
        return False

# ======================================== APP INTERFACE ========================================
def mainApp():

  customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
  customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

  app = customtkinter.CTk()  # create CTk window like you do with the Tk window
  app.geometry("1600x1600")
  app.title("StarLeague")
  #app.iconbitmap(default="pico.ico") #may only work on windows 

  headerlabel = customtkinter.CTkLabel(master=app, text="StarLeague", text_font=("Waltograph UI", 32))
  headerlabel.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)

  faqLabel = customtkinter.CTkLabel(master=app, text="FAQ: https://thepegasus.notion.site/FAQ-85ba8703cfb449ebac096ffbc40a1b95")
  faqLabel.place(relx=0.8, rely=0.8, anchor=tkinter.CENTER)

  #Sliders
  slider_var = tkinter.IntVar(value=10)
  def slider_event(valueSlider):
    sliderlabel2 = customtkinter.CTkLabel(master=app, text= str(int(valueSlider)))
    sliderlabel2.place(relx=0.8, rely=0.2, anchor=tkinter.CENTER)

    #print(valueSlider)

  sliderlabel1 = customtkinter.CTkLabel(master=app, text="Speed/Performance (results may vary)")
  sliderlabel1.place(relx=0.2, rely=0.2, anchor=tkinter.CENTER)

  slider = customtkinter.CTkSlider(
                                    master=app, 
                                    variable = slider_var,
                                    from_=1, 
                                    to=30, 
                                    number_of_steps=30, 
                                    button_color="purple", 
                                    command=slider_event
                                  )

  slider.place(relx=0.5, rely=0.2, anchor=tkinter.CENTER)

  #slider for number of seconds to pad detection instances
  paddingSlider_var = tkinter.IntVar(value=1)
  def paddingSlider_event(valueSlider):
    paddingSliderlabel2 = customtkinter.CTkLabel(master=app, text= str(int(valueSlider)))
    paddingSliderlabel2.place(relx=0.8, rely=0.3, anchor=tkinter.CENTER)

    #print(valueSlider)

  paddingSliderlabel1 = customtkinter.CTkLabel(master=app, text="Padding Per Frame + or -")
  paddingSliderlabel1.place(relx=0.2, rely=0.3, anchor=tkinter.CENTER)

  paddingSlider = customtkinter.CTkSlider(
                                    master=app, 
                                    variable = paddingSlider_var,
                                    from_=0, 
                                    to=10, 
                                    number_of_steps=30, 
                                    button_color="purple", 
                                    command=paddingSlider_event
                                  )

  paddingSlider.place(relx=0.5, rely=0.3, anchor=tkinter.CENTER)

  #Dropdown menu
  comboboxlabel1 = customtkinter.CTkLabel(master=app, text=" I Should Look For What?")
  comboboxlabel1.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)

  combobox_var = customtkinter.StringVar(value="People Present")  # set initial value

  def combobox_callback(choice):
    comboboxGlobal = choice #tkinter UI sometimes forces you to use global vars, we don't yet but may need to make this global later 
    #print("combobox dropdown clicked:", choice)

  combobox = customtkinter.CTkComboBox(master=app,
                                      values=["People Present", "Basketball", "Soccer"],
                                      text_color="black",
                                      state="readonly",
                                      command=combobox_callback,
                                      variable=combobox_var)
  combobox.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

  #CTkButtons

  runninglabel = customtkinter.CTkLabel(master=app, text="It might take a few minutes depending on the length of your video, please hang on :)", text_font=("Waltograph UI", 16))
  runninglabel.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER)

  def button_function():
    filetypes = (
          ('video files', '*.mp4 *.mov *.avi *.ogg *.mpg *.webm'),
          ('All files', '*.*')
      )

    videoFilename = filedialog.askopenfilename(
        title='Choose a video',
        filetypes=filetypes)

    if str(combobox_var.get()) == "People Present":
      activityOption = "Human"
    elif str(combobox_var.get()) == "Basketball": #As system expands or changes these things may become disjoint as per "People Present"/"Human" instance, so handle each explicitly
      activityOption = "Basketball"
    elif str(combobox_var.get()) == "Soccer":
      activityOption = "Soccer"
    else: #error case
      activityOption = "Human"

    outputVideoName = videoFilename.split(".mp4")[0] + f"-{activityOption}edit.mp4" #f"{activityOption}-editedVideo.mp4" #videoFilename.split(".mp4")[0] + "-edit.mp4"

    #from pegasusAlgos import PegasusMain, PegasusPoseUtils, PegasusVideoUtils, PegasusImageUtils
 try:
      pegasusMain = PegasusMain()
      pegasusMain.runMain(
        videoFilePath = videoFilename, 
        activitySelection = activityOption, 
        complexityReducer=int( slider_var.get() ), 
        numSecondsPaddingToApply=int( paddingSlider_var.get() ),
        outputVideoPath=outputVideoName
        )
      
      successMessage = f"Video successfully edited and saved to the location of your input video!"
    except:
      successMessage = "Sorry something went wrong, please try again. Adjust settings, reload the app, or see our FAQ if issues persist."
  
    successlabel = customtkinter.CTkLabel(master=app, text=successMessage, text_font=("Waltograph UI", 12))
    successlabel.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)

    #print("button pressed, options: ", activityOption, int( slider_var.get() ), int( paddingSlider_var.get() ), videoFilename)

  button = customtkinter.CTkButton(master=app, text="Transform Video of Choice", fg_color="purple", hover_color="white", command=button_function)
  button.place(relx=0.2, rely=0.7, anchor=tkinter.CENTER)

  endButton = customtkinter.CTkButton(master=app, text="Close", fg_color="red", hover_color="white", command=app.destroy)
  endButton.place(relx=0.8, rely=0.7, anchor=tkinter.CENTER)

  app.mainloop()



# =================================================================================================================
try:
    if checkGumroadLicenseValidity( numberOfUsesAllowedPerYear=1000 ):
      mainApp()
    else:
      tk_write("License Key Invalid. Please check that your license key is valid.")
except:
  tk_write("Couldn't connect.  Please check that your license key is valid and that you have a stable internet connection.")
