// Esperar a que carregui tot
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const botonEntrar = document.getElementById('entrar-portal');
    const botonTornar = document.getElementById('tornar-inici');
    const paginaPrincipal = document.getElementById('pagina-principal');
    const paginaPortal = document.getElementById('pagina-portal');

    // STREAMS
    const streamReal = document.getElementById('stream-real');
    const streamVirtual = document.getElementById('stream-virtual');
    
    // IMPORTANTE: Cambia estas URLs por la que te da LocalTunnel
    const STREAM_REAL_URL = "https://mi-portal.loca.lt/video_feed_real";
    const STREAM_VIRTUAL_URL = "https://mi-portal.loca.lt/video_feed_virtual";
    
    let streamsActius = false;

    // Funcio per a entrar al portal
    botonEntrar.addEventListener('click', function() {
        // Ocultem la pagina inicial i mostrem la oculta
        paginaPrincipal.classList.remove('pagina-activa');
        paginaPrincipal.classList.add('pagina-oculta');
        
        // La pagina inicial passa a anar on la pagina oculta
        paginaPortal.classList.remove('pagina-oculta');
        paginaPortal.classList.add('pagina-activa');
        iniciarStreams();
    });
    
    // Función para volver al inicio
    botonTornar.addEventListener('click', function() {
        // Ocultar página del portal
        paginaPortal.classList.remove('pagina-activa');
        paginaPortal.classList.add('pagina-oculta');
        
        // Mostrar página principal
        paginaPrincipal.classList.remove('pagina-oculta');
        paginaPrincipal.classList.add('pagina-activa');
        aturarStreams();
    });

    function iniciarStreams(){
        if(!streamsActius){
            console.log("Iniciando streams...");
            
            // Añadir timestamp para evitar cache
            const timestamp = new Date().getTime();
            streamReal.src = `${STREAM_REAL_URL}?t=${timestamp}`;
            streamVirtual.src = `${STREAM_VIRTUAL_URL}?t=${timestamp}`;

            streamReal.classList.remove('error-stream');
            streamVirtual.classList.remove('error-stream');

            streamsActius = true;
        }
    }

    function aturarStreams() {
        streamReal.src = '';
        streamVirtual.src = '';
        streamsActius = false;
    }

    streamReal.addEventListener('error', function() {
        console.error('Error cargando stream real');
        streamReal.classList.add('error-stream');
    });

    streamVirtual.addEventListener('error', function() {
        console.error('Error cargando stream virtual');
        streamVirtual.classList.add('error-stream');
    });

    // Manejar cuando los streams se cargan correctamente
    streamReal.addEventListener('load', function() {
        console.log('Stream real cargado correctamente');
        streamReal.classList.remove('error-stream');
    });

    streamVirtual.addEventListener('load', function() {
        console.log('Stream virtual cargado correctamente');
        streamVirtual.classList.remove('error-stream');
    });
});