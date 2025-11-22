// Esperar a que carregui tot
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const botonEntrar = document.getElementById('entrar-portal');
    const botonTornar = document.getElementById('tornar-inici');
    const paginaPrincipal = document.getElementById('pagina-principal');
    const paginaPortal = document.getElementById('pagina-portal');

    // Elementos de video y canvas
    const streamReal = document.getElementById('stream-real');
    const streamVirtual = document.getElementById('stream-virtual');
    const realCanvas = document.getElementById('real-canvas');
    const virtualCanvas = document.getElementById('stream-virtual');
    
    let streamsActius = false;
    let videoStream = null;
    let processingInterval = null;

    // Esperar a que OpenCV cargue
    let cvReady = false;
    cv['onRuntimeInitialized'] = function() {
        cvReady = true;
        console.log('OpenCV loaded');
    };

    // Funcio per a entrar al portal
    botonEntrar.addEventListener('click', function() {
        paginaPrincipal.classList.remove('pagina-activa');
        paginaPrincipal.classList.add('pagina-oculta');
        
        paginaPortal.classList.remove('pagina-oculta');
        paginaPortal.classList.add('pagina-activa');
        
        iniciarStreams();
    });
    
    // Función para volver al inicio
    botonTornar.addEventListener('click', function() {
        paginaPortal.classList.remove('pagina-activa');
        paginaPortal.classList.add('pagina-oculta');
        
        paginaPrincipal.classList.remove('pagina-oculta');
        paginaPrincipal.classList.add('pagina-activa');
        
        aturarStreams();
    });

    async function iniciarStreams() {
        if (!streamsActius) {
            try {
                // Acceder a la cámara
                videoStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                
                streamReal.srcObject = videoStream;
                streamsActius = true;
                
                // Configurar canvases
                realCanvas.width = virtualCanvas.width = 640;
                realCanvas.height = virtualCanvas.height = 480;
                
                // Esperar a que OpenCV esté listo
                const checkOpenCV = setInterval(() => {
                    if (cvReady) {
                        clearInterval(checkOpenCV);
                        iniciarProcesamiento();
                    }
                }, 100);
                
            } catch (error) {
                console.error('Error accediendo a la cámara:', error);
                alert('No se pudo acceder a la cámara. Asegúrate de permitir el acceso.');
            }
        }
    }

    function iniciarProcesamiento() {
        if (processingInterval) clearInterval(processingInterval);
        
        processingInterval = setInterval(() => {
            if (cvReady && streamsActius) {
                procesarFrame();
            }
        }, 1000 / 15); // 15 FPS
    }

    function procesarFrame() {
        const ctxReal = realCanvas.getContext('2d');
        const ctxVirtual = virtualCanvas.getContext('2d');
        
        // Dibujar frame actual en canvas real
        ctxReal.drawImage(streamReal, 0, 0, realCanvas.width, realCanvas.height);
        
        // Obtener imagen del canvas
        const imageData = ctxReal.getImageData(0, 0, realCanvas.width, realCanvas.height);
        
        // Crear matriz OpenCV desde ImageData
        const src = cv.matFromImageData(imageData);
        const gray = new cv.Mat();
        const fgMask = new cv.Mat();
        const maskThresh = new cv.Mat();
        const kernel = new cv.Mat();
        const maskEroded = new cv.Mat();
        
        try {
            // Convertir a escala de grises
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            
            // Aplicar sustracción de fondo (simulada con threshold)
            cv.threshold(gray, fgMask, 100, 255, cv.THRESH_BINARY);
            
            // Aplicar threshold adicional
            cv.threshold(fgMask, maskThresh, 120, 255, cv.THRESH_BINARY);
            
            // Operaciones morfológicas
            kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
            cv.morphologyEx(maskThresh, maskEroded, cv.MORPH_OPEN, kernel);
            
            // Encontrar contornos
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(maskEroded, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            // Filtrar contornos por área
            const result = new cv.Mat();
            cv.cvtColor(gray, result, cv.COLOR_GRAY2RGBA);
            
            for (let i = 0; i < contours.size(); ++i) {
                const cnt = contours.get(i);
                const area = cv.contourArea(cnt);
                if (area > 500) {
                    const color = new cv.Scalar(0, 255, 0, 255);
                    cv.drawContours(result, contours, i, color, 2);
                }
            }
            
            // Mostrar resultado en canvas virtual
            cv.imshow(virtualCanvas, result);
            
            // Liberar memoria
            src.delete(); gray.delete(); fgMask.delete(); 
            maskThresh.delete(); kernel.delete(); maskEroded.delete();
            contours.delete(); hierarchy.delete(); result.delete();
            
        } catch (error) {
            console.error('Error en procesamiento:', error);
        }
    }

    function aturarStreams() {
        if (processingInterval) {
            clearInterval(processingInterval);
            processingInterval = null;
        }
        
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }
        
        streamReal.srcObject = null;
        streamsActius = false;
    }
});