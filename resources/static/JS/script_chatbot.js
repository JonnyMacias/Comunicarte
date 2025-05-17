const video = document.getElementById('video_chat');

window.onload = async function() {
    startVideo();
  };

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