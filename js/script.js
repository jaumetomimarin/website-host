document.addEventListener('DOMContentLoaded', function() {

    const botonEntrar = document.getElementById('entrar-portal');
    const botonTornar = document.getElementById('tornar-inici');
    const paginaPrincipal = document.getElementById('pagina-principal');
    const paginaPortal = document.getElementById('pagina-portal');

    const streamReal = document.getElementById('stream-real');
    const streamVirtual = document.getElementById('stream-virtual');

    // Cambia ESTO por tu localtunnel
    const BASE_URL = "https://mi-portal.loca.lt";

    const STREAM_REAL_URL = BASE_URL + "/video_feed_real";
    const STREAM_VIRTUAL_URL = BASE_URL + "/video_feed_virtual";

    let streamsActius = false;


    botonEntrar.addEventListener('click', function() {

        paginaPrincipal.classList.remove('pagina-activa');
        paginaPrincipal.classList.add('pagina-oculta');

        paginaPortal.classList.remove('pagina-oculta');
        paginaPortal.classList.add('pagina-activa');

        iniciarStreams();
    });


    botonTornar.addEventListener('click', function() {

        paginaPortal.classList.remove('pagina-activa');
        paginaPortal.classList.add('pagina-oculta');

        paginaPrincipal.classList.remove('pagina-oculta');
        paginaPrincipal.classList.add('pagina-activa');

        aturarStreams();
    });


    function iniciarStreams(){
        if(!streamsActius){
            streamReal.src = STREAM_REAL_URL;
            streamVirtual.src = STREAM_VIRTUAL_URL;
            streamsActius = true;
        }
    }

    function aturarStreams() {
        streamReal.src = '';
        streamVirtual.src = '';
        streamsActius = false;
    }

});
