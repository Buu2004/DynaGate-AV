const carouselData={
audio:{
conditions:["SNR = -10 dB","SNR = 0 dB","SNR = 10 dB"],
baseline:["79.4%","86.1%","88.2%"],
dynagate:["85.5%","90.1%","91.7%"],
improvement:["+6.1 percentage points","+4.0 percentage points","+3.5 percentage points"],
confusion:["assets/audio_noise_1.png","assets/audio_noise_2.png","assets/audio_noise_3.png"],
routing:["assets/audio_routing_1.png","assets/audio_routing_2.png","assets/audio_routing_3.png"]
},
video:{
conditions:["σ = 0.10","σ = 0.15","σ = 0.20"],
baseline:["89.9%","85.3%","77.8%"],
dynagate:["92.0%","89.4%","82.6%"],
improvement:["+2.1 percentage points","+4.1 percentage points","+4.8 percentage points"],
confusion:["assets/video_noise_1.png","assets/video_noise_2.png","assets/video_noise_3.png"],
routing:["assets/video_routing_1.png","assets/video_routing_2.png","assets/video_routing_3.png"]
}
};

const initModalityCarousel=(element,key)=>{
const data=carouselData[key];
let index=0;
const prev=element.querySelector(".carousel-prev");
const next=element.querySelector(".carousel-next");
const condition=element.querySelector(".noise-condition");
const baseline=element.querySelector(".summary-baseline");
const dynagate=element.querySelector(".summary-dynagate");
const improvement=element.querySelector(".summary-improvement");
const confusion=element.querySelector(".noise-confusion-image");
const routing=element.querySelector(".routing-image");

const render=()=>{
condition.textContent=data.conditions[index];
baseline.textContent=data.baseline[index];
dynagate.textContent=data.dynagate[index];
improvement.textContent=data.improvement[index];
confusion.src=data.confusion[index];
routing.src=data.routing[index];
confusion.alt=`Confusion matrices under ${data.conditions[index]}`;
routing.alt=`CNN-gate expert selection distribution under ${data.conditions[index]}`;
};

prev.addEventListener("click",()=>{
index=(index-1+data.conditions.length)%data.conditions.length;
render();
});

next.addEventListener("click",()=>{
index=(index+1)%data.conditions.length;
render();
});

render();
};

document.querySelectorAll('[data-carousel]').forEach(element=>{
initModalityCarousel(element,element.dataset.carousel);
});

document.addEventListener("DOMContentLoaded",()=>{
const elements=document.querySelectorAll(".info-card,.metric-card,.dataset-stat,.full-figure-card,.noise-card,.method-row");
const observer=new IntersectionObserver(entries=>{
entries.forEach(entry=>{
if(entry.isIntersecting){
entry.target.classList.add("visible");
observer.unobserve(entry.target);
}
});
},{threshold:.08});

elements.forEach(element=>{
element.classList.add("scroll-hidden");
observer.observe(element);
});
});
