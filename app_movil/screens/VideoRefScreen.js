import { Ionicons } from "@expo/vector-icons";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { Video } from "expo-av";
import { SERVER_URL } from "../config";

// Mapeo ESTÁTICO de videos disponibles en assets/videos (require estático necesario para bundler)
const AVAILABLE_VIDEOS = {
  "escoliosis lumbar": [
    { name: "Ambas rodillas al pecho.mp4", src: require('../assets/videos/escoliosis lumbar/Ambas rodillas al pecho.mp4') },
    { name: "Cubito supino.mp4", src: require('../assets/videos/escoliosis lumbar/Cubito supino.mp4') },
    { name: "Perro de caza.mp4", src: require('../assets/videos/escoliosis lumbar/Perro de caza.mp4') },
    { name: "Plancha lateral.mp4", src: require('../assets/videos/escoliosis lumbar/Plancha lateral.mp4') },
    { name: "Postura del Avion.mp4", src: require('../assets/videos/escoliosis lumbar/Postura del Avion.mp4') },
    { name: "Puente.mp4", src: require('../assets/videos/escoliosis lumbar/Puente.mp4') },
    { name: "Rodilla al pecho.mp4", src: require('../assets/videos/escoliosis lumbar/Rodilla al pecho.mp4') },
  ],
  "espondilolisis": [
    { name: "Abdominales.mp4", src: require('../assets/videos/espondilolisis/Abdominales.mp4') },
    { name: "Ambas rodillas al pecho.mp4", src: require('../assets/videos/espondilolisis/Ambas rodillas al pecho.mp4') },
    { name: "Perro de caza.mp4", src: require('../assets/videos/espondilolisis/Perro de caza.mp4') },
    { name: "Plancha sobre codos.mp4", src: require('../assets/videos/espondilolisis/Plancha sobre codos.mp4') },
    { name: "Puente.mp4", src: require('../assets/videos/espondilolisis/Puente.mp4') },
    { name: "Rodilla al pecho.mp4", src: require('../assets/videos/espondilolisis/Rodilla al pecho.mp4') },
  ],
  "hernia de disco lumbar": [
    { name: "Ambas rodillas al pecho.mp4", src: require('../assets/videos/hernia de disco lumbar/Ambas rodillas al pecho.mp4') },
    { name: "El perro y gato.mp4", src: require('../assets/videos/hernia de disco lumbar/El perro y gato.mp4') },
    { name: "En cuatro puntos.mp4", src: require('../assets/videos/hernia de disco lumbar/En cuatro puntos.mp4') },
    { name: "Piernas al abdomen.mp4", src: require('../assets/videos/hernia de disco lumbar/Piernas al abdomen.mp4') },
    { name: "Posicion de cobra.mp4", src: require('../assets/videos/hernia de disco lumbar/Posicion de cobra.mp4') },
    { name: "Posicion de esfinge.mp4", src: require('../assets/videos/hernia de disco lumbar/Posicion de esfinge.mp4') },
    { name: "Rodilla al pecho.mp4", src: require('../assets/videos/hernia de disco lumbar/Rodilla al pecho.mp4') },
  ],
  "lumbalgia mecánica inespecífica": [
    { name: "Abdominales (pierna contraria).mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Abdominales (pierna contraria).mp4') },
    { name: "Abdominales (pierna del mismo lado).mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Abdominales (pierna del mismo lado).mp4') },
    { name: "Abdominales.mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Abdominales.mp4') },
    { name: "El perro y gato.mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/El perro y gato.mp4') },
    { name: "Extension de la columna.mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Extension de la columna.mp4') },
    { name: "Inclinacion pelvica de pie.mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Inclinacion pelvica de pie.mp4') },
    { name: "Perro de caza.mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Perro de caza.mp4') },
    { name: "Plancha lateral.mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Plancha lateral.mp4') },
    { name: "Puente.mp4", src: require('../assets/videos/lumbalgia mecánica inespecífica/Puente.mp4') },
  ],
};

export default function VideoRefScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [step, setStep] = useState("select_condition"); // "select_condition" | "select_video" | "recording"
  const [selectedCondition, setSelectedCondition] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [mensajePostura, setMensajePostura] = useState("");
  const [showFeedback, setShowFeedback] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [facing, setFacing] = useState("front");
  const [videoCurrentTime, setVideoCurrentTime] = useState(0); // ⭐ Sincronización: tiempo actual del video
  const [videoLoaded, setVideoLoaded] = useState(false); // ⭐ Indica si video está listo
  const [isVideoExpanded, setIsVideoExpanded] = useState(false); // ⭐ Controla modo expandido del video
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const syncIntervalRef = useRef(null); // ⭐ Referencia al intervalo de sincronización

  useEffect(() => {
    if (!permission) {
      requestPermission();
    }
  }, [permission]);

  // ⭐ Efecto de sincronización de tiempo del video con servidor
  useEffect(() => {
    if (step === "recording" && videoLoaded && videoCurrentTime !== null) {
      // Limpiar intervalo anterior si existe
      if (syncIntervalRef.current) {
        clearInterval(syncIntervalRef.current);
      }

      // Emitir tiempo cada 250ms al servidor (igual que en web)
      syncIntervalRef.current = setInterval(async () => {
        try {
          await fetch(`${SERVER_URL}/api/sync_reference_time`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ current_time: videoCurrentTime }),
          });
        } catch (err) {
          // Silenciar errores de sincronización (no es crítico)
          console.log("Sync error (non-blocking):", err.message);
        }
      }, 250);
    }

    return () => {
      if (syncIntervalRef.current) {
        clearInterval(syncIntervalRef.current);
      }
    };
  }, [step, videoLoaded, videoCurrentTime]);

  // Captura periódica y evaluación remota (SIN mostrar capturas, solo feedback)
  useEffect(() => {
    let captureInterval;

    const doCapture = async () => {
      try {
        if (cameraRef.current && cameraRef.current.takePictureAsync && !isEvaluating) {
          setIsEvaluating(true);
          let photo = null;
          try {
            photo = await cameraRef.current.takePictureAsync({
              quality: 0.3, // Menor calidad para más velocidad
              base64: true,
              skipProcessing: true,
            });
          } catch (captureErr) {
            console.log("capture attempt failed, retrying:", captureErr);
            // retry once after short delay
            await new Promise((res) => setTimeout(res, 300));
            try {
              photo = await cameraRef.current.takePictureAsync({
                quality: 0.3,
                base64: true,
                skipProcessing: true,
              });
            } catch (secondErr) {
              console.log("second capture attempt failed:", secondErr);
              setMensajePostura("No se pudo capturar la imagen");
              setShowFeedback(true);
              setTimeout(() => setShowFeedback(false), 1500);
              setIsEvaluating(false);
              return; // abort this cycle
            }
          }

          if (photo && photo.base64) {
            try {
              const resp = await fetch(`${SERVER_URL}/api/evaluate_frame`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  image: `data:image/jpeg;base64,${photo.base64}`,
                }),
              });

              const j = await resp.json();
              if (j && j.success) {
                const serverFeedback = (j.feedback || "Evaluado");
                const serverMetrics = j.metrics || {};

                // Heurística cliente: si se reportan métricas de distancia, confiar en ellas
                let isGood = false;
                if (
                  serverMetrics &&
                  typeof serverMetrics.avg_distance === "number"
                ) {
                  // Umbral empírico: avg_distance pequeño => postura correcta
                  isGood = serverMetrics.avg_distance < 0.05; // ajustar si es necesario
                } else {
                  // Fallback a la retroalimentación del servidor
                  isGood = serverFeedback.toLowerCase().includes("bien");
                }

                const displayText = isGood
                  ? "Bien — ¡Buen movimiento!"
                  : "Mal — Ajusta la postura";

                setMensajePostura(displayText);
                setMetrics(serverMetrics);
              } else {
                setMensajePostura("Error de evaluación");
              }
            } catch (err) {
              console.log("Network eval error", err);
              setMensajePostura("Sin conexión");
            }

            setShowFeedback(true);
            setTimeout(() => setShowFeedback(false), 2000); // Mostrar feedback 2s
            setIsEvaluating(false);
          }
        }
      } catch (e) {
        console.log("capture error", e);
        setIsEvaluating(false);
      }
    };

    if (step === "recording" && isCameraReady) {
      doCapture();
      captureInterval = setInterval(doCapture, 1500); // Evaluar cada 1.5s
    }

    return () => {
      if (captureInterval) clearInterval(captureInterval);
    };
  }, [step, isCameraReady, isEvaluating]);

  if (!permission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.text}>Verificando permisos de cámara...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.text}>Se necesita acceso a la cámara</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Dar permiso</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Paso 1: Seleccionar condición médica
  if (step === "select_condition") {
    return (
      <View style={styles.previewContainer}>
        <Text style={styles.title}>Selecciona una condición médica</Text>
        <ScrollView style={styles.listContainer}>
          {Object.keys(AVAILABLE_VIDEOS).map((condition) => (
            <TouchableOpacity
              key={condition}
              style={styles.conditionButton}
              onPress={() => {
                setSelectedCondition(condition);
                setStep("select_video");
              }}
            >
              <Text style={styles.conditionButtonText}>{condition}</Text>
              <Ionicons name="chevron-forward" size={24} color="#7a8ce2" />
            </TouchableOpacity>
          ))}
        </ScrollView>
        <TouchableOpacity
          style={styles.backBtn}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.backText}>⬅ Volver</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Paso 2: Seleccionar video
  if (step === "select_video" && selectedCondition) {
    const videos = AVAILABLE_VIDEOS[selectedCondition] || [];
    return (
      <View style={styles.previewContainer}>
        <Text style={styles.title}>Elige un video de referencia</Text>
        <Text style={styles.subtitle}>{selectedCondition}</Text>
        <ScrollView style={styles.listContainer}>
          {videos.map((video, idx) => (
            <TouchableOpacity
              key={idx}
              style={styles.videoButton}
              onPress={() => {
                setSelectedVideo(video);
                setStep("recording");
              }}
            >
              <Ionicons name="play-circle" size={32} color="#7a8ce2" />
              <Text style={styles.videoButtonText}>{video.name}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
        <TouchableOpacity
          style={styles.backBtn}
          onPress={() => {
            setSelectedCondition(null);
            setStep("select_condition");
          }}
        >
          <Text style={styles.backText}>⬅ Volver</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Paso 3: Grabación y evaluación en tiempo real
  if (step === "recording" && selectedCondition && selectedVideo) {

    return (
      <View style={styles.container}>
        <CameraView
          style={styles.camera}
          ref={cameraRef}
          facing={facing}
          onCameraReady={() => setIsCameraReady(true)}
        />

        {/* ⭐ Video de referencia en overlay (PiP) - Versión compacta */}
        {!isVideoExpanded && (
          <TouchableOpacity
            style={styles.videoPiPContainer}
            onPress={() => setIsVideoExpanded(true)}
          >
            <Video
              ref={videoRef}
              source={selectedVideo?.src}
              style={styles.videoPiP}
              rate={1.0}
              volume={0.5}
              resizeMode="cover"
              shouldPlay
              useNativeControls={false}
              isLooping={true}
              onProgress={(e) => {
                setVideoCurrentTime(e.currentTime);
              }}
              onLoad={() => {
                setVideoLoaded(true);
              }}
            />
            <Text style={styles.videoPiPTime}>{Math.floor(videoCurrentTime)}s</Text>
            <Ionicons
              name="expand"
              size={20}
              color="#fff"
              style={styles.expandIcon}
            />
          </TouchableOpacity>
        )}

        {/* ⭐ Modal expandido del video (pantalla completa) */}
        {isVideoExpanded && (
          <Modal
            visible={isVideoExpanded}
            transparent={false}
            animationType="slide"
            onRequestClose={() => setIsVideoExpanded(false)}
          >
            <View style={styles.expandedVideoContainer}>
              <Video
                ref={videoRef}
                source={selectedVideo?.src}
                style={styles.expandedVideo}
                rate={1.0}
                volume={1.0}
                resizeMode="contain"
                shouldPlay
                useNativeControls={true}
                isLooping={true}
                onProgress={(e) => {
                  setVideoCurrentTime(e.currentTime);
                }}
                onLoad={() => {
                  setVideoLoaded(true);
                }}
              />
              <TouchableOpacity
                style={styles.closeExpandedButton}
                onPress={() => setIsVideoExpanded(false)}
              >
                <Ionicons name="close" size={32} color="#fff" />
              </TouchableOpacity>
              <Text style={styles.expandedVideoTime}>
                {selectedCondition} - {Math.floor(videoCurrentTime)}s
              </Text>
            </View>
          </Modal>
        )}

        {/* Encabezado con botón atrás */}
        <View style={styles.overlayTop}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => {
              setStep("select_video");
              setSelectedVideo(null);
              setVideoLoaded(false);
              setVideoCurrentTime(0);
              setIsVideoExpanded(false);
            }}
          >
            <Ionicons name="arrow-back" size={28} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* Información y cambio de cámara */}
        <View style={styles.overlayTopRight}>
          <View style={styles.infoBox}>
            <Text style={styles.infoText}>{selectedCondition}</Text>
          </View>
        </View>

        {/* Botón flip cámara */}
        <View style={styles.overlayBottomRight}>
          <TouchableOpacity
            style={styles.flipButton}
            onPress={() =>
              setFacing((prev) => (prev === "front" ? "back" : "front"))
            }
          >
            <Ionicons name="camera-reverse" size={30} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* Feedback flotante - FEEDBACK VISUAL MEJORADO */}
        {showFeedback && (
          <View
            style={[
              styles.feedbackContainer,
              {
                backgroundColor:
                  mensajePostura.includes("Bien") || mensajePostura.includes("✓")
                    ? "rgba(76, 175, 80, 0.95)"
                    : "rgba(255, 107, 107, 0.95)",
              },
            ]}
          >
            {/* Emoji indicador */}
            <Text style={styles.feedbackEmoji}>
              {mensajePostura.includes("Bien") || mensajePostura.includes("✓") ? "✅" : "❌"}
            </Text>
            
            {/* Texto feedback */}
            <Text style={styles.feedbackText}>{mensajePostura}</Text>
            
            {/* Métricas opcionales */}
            {metrics && (metrics.avg_distance || metrics.max_distance) && (
              <Text style={styles.metricsText}>
                Distancia: {metrics.avg_distance?.toFixed(3) || "—"}
              </Text>
            )}
          </View>
        )}

        {/* Indicador de evaluación */}
        {isEvaluating && (
          <View style={styles.evaluatingIndicator}>
            <ActivityIndicator size="small" color="#fff" />
          </View>
        )}
      </View>
    );
  }

  return (
    <View style={styles.centered}>
      <Text style={styles.text}>Cargando...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  // Selección y preview
  previewContainer: {
    flex: 1,
    backgroundColor: "#f0f3ff",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  title: {
    fontSize: 26,
    fontWeight: "800",
    color: "#4a56a6",
    marginBottom: 20,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#7a8ce2",
    marginBottom: 15,
    textAlign: "center",
  },
  listContainer: {
    width: "100%",
    maxHeight: 400,
  },
  conditionButton: {
    backgroundColor: "#fff",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingVertical: 15,
    marginVertical: 8,
    borderRadius: 12,
    borderLeftWidth: 5,
    borderLeftColor: "#7a8ce2",
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 5,
    elevation: 2,
  },
  conditionButtonText: {
    flex: 1,
    fontSize: 16,
    fontWeight: "600",
    color: "#4a56a6",
  },
  videoButton: {
    backgroundColor: "#fff",
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 12,
    marginVertical: 8,
    borderRadius: 10,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 4,
    elevation: 2,
  },
  videoButtonText: {
    fontSize: 15,
    fontWeight: "500",
    color: "#555",
    marginLeft: 15,
    flex: 1,
  },
  backBtn: {
    marginTop: 30,
    backgroundColor: "transparent",
  },
  backText: {
    color: "#7a8ce2",
    fontWeight: "700",
    fontSize: 16,
  },

  // Cámara
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  camera: {
    flex: 1,
  },
  // ⭐ Video PiP (Picture-in-Picture) - Compacto
  videoPiPContainer: {
    position: "absolute",
    top: 80,
    left: 20,
    width: 140,
    height: 200,
    borderRadius: 10,
    overflow: "hidden",
    borderWidth: 2,
    borderColor: "#7a8ce2",
    backgroundColor: "#000",
    zIndex: 10,
  },
  videoPiP: {
    width: "100%",
    height: "100%",
  },
  videoPiPTime: {
    position: "absolute",
    bottom: 8,
    right: 8,
    backgroundColor: "rgba(0,0,0,0.8)",
    color: "#fff",
    fontSize: 12,
    fontWeight: "700",
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  expandIcon: {
    position: "absolute",
    top: 8,
    right: 8,
  },
  // ⭐ Video expandido (pantalla completa)
  expandedVideoContainer: {
    flex: 1,
    backgroundColor: "#000",
    justifyContent: "center",
    alignItems: "center",
  },
  expandedVideo: {
    width: "100%",
    height: "80%",
  },
  closeExpandedButton: {
    position: "absolute",
    top: 50,
    right: 20,
    backgroundColor: "rgba(0,0,0,0.7)",
    padding: 10,
    borderRadius: 30,
    zIndex: 100,
  },
  expandedVideoTime: {
    position: "absolute",
    bottom: 20,
    left: 20,
    right: 20,
    backgroundColor: "rgba(0,0,0,0.8)",
    color: "#fff",
    fontSize: 14,
    fontWeight: "700",
    padding: 12,
    borderRadius: 8,
    textAlign: "center",
  },
  overlayTop: {
    position: "absolute",
    top: 50,
    left: 20,
    zIndex: 40,
    elevation: 40,
  },
  backButton: {
    backgroundColor: "rgba(0,0,0,0.5)",
    padding: 10,
    borderRadius: 50,
  },
  overlayTopRight: {
    position: "absolute",
    top: 60,
    right: 20,
    maxWidth: 150,
  },
  infoBox: {
    backgroundColor: "rgba(0,0,0,0.7)",
    paddingHorizontal: 15,
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: "#7a8ce2",
  },
  infoText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "700",
    marginBottom: 4,
  },
  videoNameText: {
    color: "#7a8ce2",
    fontSize: 12,
    fontWeight: "500",
  },

  // Botón flip
  overlayBottomRight: {
    position: "absolute",
    bottom: 100,
    right: 30,
  },
  flipButton: {
    backgroundColor: "rgba(0,0,0,0.6)",
    padding: 12,
    borderRadius: 50,
  },

  // Feedback - MEJORADO: más visible y llamativo
  feedbackContainer: {
    position: "absolute",
    bottom: 160,
    alignSelf: "center",
    paddingVertical: 20,
    paddingHorizontal: 40,
    borderRadius: 20,
    alignItems: "center",
    minWidth: 180,
    shadowColor: "#000",
    shadowOpacity: 0.5,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 8,
    elevation: 10,
  },
  feedbackEmoji: {
    fontSize: 48,
    marginBottom: 8,
  },
  feedbackText: {
    color: "#fff",
    fontWeight: "800",
    fontSize: 20,
    textAlign: "center",
    letterSpacing: 0.5,
  },
  metricsText: {
    color: "#fff",
    fontSize: 12,
    marginTop: 6,
    textAlign: "center",
    opacity: 0.9,
    fontWeight: "600",
  },

  // Indicador de evaluación - Discreto en la esquina
  evaluatingIndicator: {
    position: "absolute",
    bottom: 50,
    right: 30,
    backgroundColor: "rgba(122, 140, 226, 0.8)",
    padding: 10,
    borderRadius: 50,
    width: 40,
    height: 40,
    justifyContent: "center",
    alignItems: "center",
  },

  // Permisos/Errores
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f0f3ff",
    padding: 20,
  },
  text: {
    fontSize: 18,
    color: "#4a56a6",
    fontWeight: "600",
    textAlign: "center",
  },
  btn: {
    marginTop: 15,
    backgroundColor: "#4a56a6",
    paddingVertical: 10,
    paddingHorizontal: 25,
    borderRadius: 10,
  },
  btnText: {
    color: "#fff",
    fontWeight: "600",
  },
});
