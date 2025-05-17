
const mano = document.getElementById("mano");
const barraLateral = document.querySelector(".barra_lateral");
const spans = document.querySelectorAll("span");
const palanca = document.querySelector(".switch");
const circulo = document.querySelector(".circulo");
const menu = document.querySelector(".menu");
const main = document.querySelector("main");
const barraSuperior = document.querySelector(".barra_superior");

menu.addEventListener("click",()=>{
   barraLateral.classList.toggle("max-barra-lateral")
   if(barraLateral.classList.contains("max-barra-lateral")){
      menu.children[0].style.display = "none";
      menu.children[1].style.display = "block";
   }
   else{
      menu.children[0].style.display = "block";
      menu.children[1].style.display = "none";
   }
   if(window.innerWidth<=320){
      barraLateral.classList.add("mini_barra_lateral")
      main.classList.add("min_main")
      spans.forEach((span)=>{
      
      span.classList.add("oculto")
      });
      
   }
})

palanca.addEventListener("click", ()=>{
   let body= document.body;
   body.classList.toggle("dark_mode");
   circulo.classList.toggle("prendido");
});

mano.addEventListener("click", ()=>{
   barraLateral.classList.toggle("mini_barra_lateral");
   main.classList.toggle("min_main");
   barraSuperior.classList.toggle("min_barra_superior");
   spans.forEach((span) => {
      span.classList.toggle("oculto");
   });
})