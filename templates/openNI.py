from primesense import openni2

def init_openni2():
    global OPENNI2_INITIALIZED
    if not OPENNI2_INITIALIZED:
        openni_path = "C:\\Program Files\\OpenNI2\\Redist"
        openni2.initialize(openni_path)
        OPENNI2_INITIALIZED = True

def unload_openni2():
    global OPENNI2_INITIALIZED
    if OPENNI2_INITIALIZED:
        openni2.unload()
        OPENNI2_INITIALIZED = False