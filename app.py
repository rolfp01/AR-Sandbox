from flask import Flask, redirect, render_template, Response, session, url_for, jsonify

from DataCalculation import calculate2DVolume, calculateHight, detectBuildings, grayPicture
from DataRead import readDepthCamera, readLaptopCamera
from DataShow import show2DVolume, showGrayPicture, showHight, showObjects

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Füge hier deinen geheimen Schlüssel hinzu

# Definition der VideoTheme-Klasse
class VideoTheme:
    def __init__(self, index, name, funct):
        self.index = index
        self.name = name
        self.funct = funct

def gray_video():
    while True:
        cameraInput = readLaptopCamera.read_Laptop_Camera()
        if type(cameraInput) == 'NoneType':
            continue
        else: 
            calculationOutput = grayPicture.picture_In_Gray(cameraInput)
            beamerOutput = showGrayPicture.show_Gray_Picture(calculationOutput)
        yield beamerOutput
'''
def objects_video():
    #while True:
        cameraInput = readDepthCamera.read_Depth_Camera()
        #calculationOutput = detectBuildings.detect_Buildings(cameraInput)
        #beamerOutput = showObjects.show_Objects(calculationOutput)
        #yield beamerOutput

def volume_2D_video():
    #while True:
        cameraInput = readDepthCamera.read_Depth_Camera()
        #calculationOutput = calculate2DVolume.calculate_2D_Volume(cameraInput)
        #beamerOutput = show2DVolume.show_2D_Volume(calculationOutput)
        #yield beamerOutput

def hights_video():
    #while True:
        cameraInput = readDepthCamera.read_Depth_Camera()
        #calculationOutput = calculateHight.calculate_Hight(cameraInput)
        #beamerOutput = showHight.show_Hight(calculationOutput)
        #yield beamerOutput
        '''

activeVideoTheme = 0
videoThemes = [
    VideoTheme(0, "Graustufen Video", gray_video),
    VideoTheme(1, "Objekte", gray_video),
    VideoTheme(2, "2D Lautstärke", gray_video),
    VideoTheme(3, "Höhe", gray_video)
]

@app.route('/')
def index():    # Überprüfen, ob 'videoTheme' in der Session existiert
    if 'activeVideoTheme' not in session:
        session['activeVideoTheme'] = 0  # Wenn nicht, auf das erste Thema setzen

    current_theme = videoThemes[session['activeVideoTheme']].name  # Hole den aktuellen Themennamen
    return render_template('index.html', current_theme=current_theme)  # Übergib den Namen an das Template

@app.route('/video_feed')
def video_feed():
    return Response(videoThemes[session['activeVideoTheme']].funct(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/theme_Switch')
def theme_Switch():
    print("test")
    # Wechsle das Thema
    session['activeVideoTheme'] = (session['activeVideoTheme'] + 1) % len(videoThemes)  # Einfacher Wechsel der Themenindex
    print(f"Thema gewechselt zu: {videoThemes[session['activeVideoTheme']].name}")
    return redirect(url_for('index'))  # Redirect zur Startseite, um das Template mit dem neuen Thema zu laden

if __name__ == '__main__':
    app.run(debug=True)
