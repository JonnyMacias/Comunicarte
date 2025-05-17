    const botonCaptura = document.getElementById('capture');
    const video = document.getElementById('video');
    const palabra = document.getElementById('palabra');
    const caracteres = document.getElementById('caracteres');
    const next = document.getElementById('next');
    const ventana = document.getElementsByClassName('container_emergente')[0];
    const reintentar = document.getElementById('reintentar');
    const eSiguiente = document.getElementById('eSiguiente');
    const atras = document.getElementById('back');
    const statusImg = document.getElementById('statusImg');
    const mensajeV = document.getElementById('mensajeV');
    const statusen = document.getElementById('statusen');
    let capturado = true;
    let intervalo = null;
    let stream = null;
    let estaEsperando = false; 

    botonCaptura.addEventListener('click', iniciarCaptura);
    next.addEventListener('click', cargarPalabra);

    // ---------------------------------- botones de ventana emergente ----------------------------------
    reintentar.addEventListener('click', reintentarN);
    back.addEventListener('click', cerrarVentana);
    eSiguiente.addEventListener('click', SiguientePalabra);

    window.onload = async function() {
      cargarPalabra();
    };
    
    // Acceder a la cámara
    async function startVideo() {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
          video.srcObject = stream;
          video.classList.add('videoContainer');
          botonCaptura.textContent = 'Iniciar';
          botonCaptura.style.backgroundColor = '#6bea86';
        } catch (error) {
          console.error('Error al acceder a la cámara:', error);
        }
      }
  
let capturedFrames = [];
const totalFramesToCapture = 83;
let isCapturing = false;
let isWaitingForResponse = false; // Bandera para controlar el estado de espera

async function captureFrame() {
  if (!isCapturing || isWaitingForResponse) return;
  
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('2d');

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL('image/png');
  capturedFrames.push(imageData);

  if (capturedFrames.length >= totalFramesToCapture) {
    isWaitingForResponse = true; // Bloquear nuevas capturas
    await enviarFrames(capturedFrames); // Esperar la respuesta
    capturedFrames = [];
    statusen.textContent = "capturando"
    statusen.style.color = "red"; 
    
    // Esperar 2 segundos adicionales después de recibir respuesta
    await dormir(2000);
    isWaitingForResponse = false; // Reanudar captura
  }
}

async function enviarFrames(framesArray) {
  statusen.textContent = "enviando"
  statusen.style.color = "green"; 
  try {
    const response = await fetch('http://localhost:5000/api/proc_img', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ imagenes: framesArray })
    });
    
    const data = await response.json();
    console.log('Respuesta del servidor:', data);
    
    if (data.mensaje !== 'not') {
      rellenarCar(data.mensaje);
    }
  } catch (error) {
    console.error('Error al enviar imágenes:', error);
  } finally {
    // La bandera isWaitingForResponse se maneja en captureFrame
  }
}

function actionIniciar() {
  botonCaptura.textContent = 'Detener';
  botonCaptura.style.backgroundColor = '#ea736b';
  isCapturing = true;
  intervalo = setInterval(captureFrame, 33); // ~30fps
}

function actionDetener() {
  isCapturing = false;
  isWaitingForResponse = false; // Cancelar cualquier espera pendiente
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    video.classList.remove('videoContainer');
    stream = null;
  }
  clearInterval(intervalo);
  botonCaptura.textContent = 'Capturar';
  botonCaptura.style.backgroundColor = 'white';
}
      // Iniciar la cámara
    function iniciarCaptura(){
      if(botonCaptura.innerText == 'Capturar'){
        startVideo();
      }else if(botonCaptura.innerText == 'Iniciar'){
        actionIniciar();
      }else{
        actionDetener();
        botonCaptura.textContent = 'Capturar'
        botonCaptura.style.backgroundColor = 'white';
      }
    }
    

    //------------------------------------ palabras a deletrear -------------------------------
    async function cargarPalabra() {
      
      try {
        const response = await fetch('http://localhost:5000/api/getPalabra', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        const data = await response.json();
        palabra.textContent = data.palabra;
        car_(); 
      } catch (error) {
        console.error('Error al obtener la palabra:', error);
      }
    }

    function car_(){
      let caracter = "";
      for( var i = 0; i < palabra.textContent.length; i++){
        caracter += "_ ";
      }
      caracteres.textContent = caracter;
    }


    async function rellenarCar(car){
      if(estaEsperando) return;

      estaEsperando = true;
      let periso = true;
      let nPlabra = '';
      for(let char of caracteres.textContent){
        if(char == '_' && periso != false){
          periso = false;
          nPlabra += car;
        }else{
          nPlabra += char;
        }
      }
      caracteres.textContent = nPlabra;
      if(periso){
        comparar();
        actionDetener();
      }else {     
        console.log('esperar');
        //await dormir(2000); // ✅ OK
        console.log('reiniciar');
      }

      estaEsperando = false;
    }
    function normalizarTexto(texto) {
      return texto
        .toLowerCase() // convierte a minúsculas
        .normalize("NFD") // descompone acentos
        .replace(/[\u0300-\u036f]/g, "") // elimina acentos
        .replace(/\s+/g, "");
    }
    

    function comparar(){
      if(normalizarTexto(palabra.textContent) == normalizarTexto(caracteres.textContent.slice(0, -1))){
        ventana.style.display = 'flex';
        statusImg.src  = 'IMG/deletreo/bienhecho.svg';
        mensajeV.textContent = 'Bien Hecho';
      }else{
        ventana.style.display = 'flex';
        statusImg.src  = 'IMG/deletreo/sad.svg';
        mensajeV.textContent = 'Hay que practicar un poco mas...';

      }
      startVideo();
    }

    // ----------------------Funsiones de ventana emergente -----------------------
    function reintentarN(){
      car_();
      cerrarVentana();
    } 

    function SiguientePalabra(){
      cargarPalabra();
      cerrarVentana();
    }

    function cerrarVentana(){
      ventana.style.display = 'none';
    }

    function dormir(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    
      
  