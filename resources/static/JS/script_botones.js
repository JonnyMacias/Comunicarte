document.addEventListener('DOMContentLoaded', function () {
    const categorySelect = document.getElementById('categorySelect');
    const containerBotones = document.querySelector('.container_botones');
    const ventana = document.getElementsByClassName('container_emergente')[0];
    const video = document.querySelector('.videoSenal')
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.style.display = 'none';
    loadingIndicator.textContent = 'Cargando...';
    categorySelect.parentNode.appendChild(loadingIndicator);

    
    

    const categoryMapping = {
        'categoria1': 'abecedario',
        'categoria2': 'Adjetivos, pronombres, preposiciones y artículos',
        'categoria3': 'animales',
        'categoria4': 'antonimos',
        'categoria5': 'casa'
    };

    categorySelect.addEventListener('change', async function () {
        const selectedValue = this.value;

        if (!selectedValue) return;

        const folderName = categoryMapping[selectedValue];

        if (!folderName) {
            showError('Categoría no válida');
            return;
        }

        try {
            // Mostrar indicador de carga
            loadingIndicator.style.display = 'block';
            categorySelect.disabled = true;

            const response = await fetch(`/api/getCarpetas?carpeta=${encodeURIComponent(folderName)}`);

            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }

            const data = await response.json();

            // Procesar los datos recibidos
            processCarpetasData(data);

        } catch (error) {
            console.error('Error:', error);
            showError('No se pudieron cargar las carpetas. Intente nuevamente.');
        } finally {
            // Ocultar indicador de carga
            loadingIndicator.style.display = 'none';
            categorySelect.disabled = false;
        }
    });

    function processCarpetasData(data) {
        containerBotones.innerHTML = '';
        data.carpeta.forEach((item) => {
            // Crear el elemento div con clase 'items'
            const nuevoItem = document.createElement('div');
            nuevoItem.className = 'items';

            // Crear el contenido HTML del item
            nuevoItem.innerHTML = `
            <img src="IMG/Logo.png" alt="${item}">
            <h3>${item}</h3>`;
            // Agregar evento click
            nuevoItem.addEventListener('click', async () => {
                try {
                    const nombreContenido = item;

                    // Hacer petición a la API
                    const response = await fetch(`/api/getContenido?nombre=${encodeURIComponent(nombreContenido)}`);

                    if (!response.ok) {
                        throw new Error('Error en la respuesta');
                    }

                    const data = await response.json();
                    console.log(data)
                    // Mostrar el contenido devuelto
                    mostrarContenido(data);

                } catch (error) {
                    console.error('Error:', error);
                    nuevoItem.innerHTML = `
          <img src="IMG/Logo.png" alt="${item}">
          <h3>${item}</h3>
          <p class="error">Error al cargar</p>
        `;
                }
            });

            // Agregar el nuevo item al contenedor
            containerBotones.appendChild(nuevoItem);
        })
    }

    function showError(message) {
        // Puedes mejorar esto con un modal o notificación bonita
        alert(`Error: ${message}`);
    }


    // Función para mostrar el contenido multimedia
    function mostrarContenido(data) {
        console.log(data.ruta_completa)
        let contenidoHTML = '';

        if (data.tipo === 'video') {
            contenidoHTML = `
            <img id="back" class="emergente_back" src="IMG/deletreo/back.svg" alt="">
            <video controls autoplay class="videoSenal">
            <source src="${data.url}" type="video/mp4">
        Tu navegador no soporta videos HTML5
      </video>
      <h3>${data.nombre.split('_')[0]}</h3>
    `;
        } else if (data.tipo === 'imagen') {
            contenidoHTML = `
             <img id="back" class="emergente_back" src="IMG/deletreo/back.svg" alt="">
      <img src="${data.url}" alt="${data.nombre} class="imgSena">
      <h3>${data.nombre.split('_')[0]}</h3>
    `;
        } else {
            contenidoHTML = `
      <img src="IMG/Logo.png" alt="${data.nombre || 'Contenido'} class="imgSena"">
      <h3>${data.nombre.split('_')[0] || 'Contenido no disponible'}</h3>
    `;
        }

        ventana.innerHTML = contenidoHTML;
        ventana.style.display = 'flex';
        const atras = document.getElementById('back');
        atras.addEventListener('click', cerrarVentana);
    }

    function cerrarVentana() {
        ventana.style.display = 'none';
        video.innerHTML = "";
    }

    document.getElementById('searchButton').addEventListener('click', () => {
    const input = document.getElementById('searchInput');
    const valor = input.value.trim();

    if (valor === "") {
        alert("Por favor ingresa un término de búsqueda.");
        return;
    }

    // Codifica el valor para que sea seguro en una URL
    const url = `api/getContenido?nombre=${encodeURIComponent(valor)}`;

    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error("Error en la búsqueda");
            }
            return response.json();
        })
        .then(data => {
            console.log("Resultado:", data);
            mostrarContenido(data);
            // Aquí puedes mostrar los datos en el DOM
        })
        .catch(error => {
            console.error("Error:", error);
        });
});
});



