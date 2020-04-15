
$(function () {
    console.log('message');
    var lyric_breaked_line = $("#lyrics")[0].textContent.replace(/\n/g, '<li>');
    $("#lyrics").html(lyric_breaked_line);
});

//播放控制
var myAudio = $("audio")[0];
// 播放/暂停控制
$("#btn1").click(function(){
    if (myAudio.paused) {
        play()
    } else {
        pause()
    }
});
// 频道切换
function play(){
    myAudio.play();
    $('#btn1').removeClass('fa-play').addClass('fa-pause');
}
function pause(){
    myAudio.pause();
    $('#btn1').removeClass('fa-pause').addClass('fa-play');
}
setInterval(present,500)    //每0.5秒计算进度条长度
$(".basebar").mousedown(function(ev){  //拖拽进度条控制进度
    var posX = ev.clientX;
    var targetLeft = $(this).offset().left;
    var percentage = (posX - targetLeft)/400*100;
    myAudio.currentTime = myAudio.duration * percentage/100;
});
function present(){
    var length = myAudio.currentTime/myAudio.duration*100;
    $('.progressbar').width(length+'%');//设置进度条长度
    //自动下一曲
    if(myAudio.currentTime == myAudio.duration){
    $('#btn1').removeClass('fa-pause').addClass('fa-play');
    }
}