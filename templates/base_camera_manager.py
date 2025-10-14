# ============================================================================
# Kamera-Manager-Klasse
# ============================================================================

class BaseCameraManager:
    """Basis-Klasse für alle Kamera-Manager"""
    
    def __init__(self):
        self.depth_scale = 1
        self.baseline_distance = None
    
    def start(self):
        """Startet die Kamera - muss von Unterklassen implementiert werden"""
        raise NotImplementedError
    
    def read_frame(self):
        """Liest ein Frame - muss von Unterklassen implementiert werden"""
        raise NotImplementedError
    
    def stop(self):
        """Stoppt die Kamera - muss von Unterklassen implementiert werden"""
        raise NotImplementedError
    