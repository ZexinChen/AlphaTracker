var v = document.getElementById('video');

var play_button = document.getElementById('play');

var back_button = document.getElementById('back_button');
back_button.onclick = function(){
    v.currentTime = v.currentTime-1/framerate;
    range.value = v.currentTime;
    chart1.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    chart2.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    draw_mice();
};

var forward_button = document.getElementById('forward_button');
forward_button.onclick = function(){
    v.currentTime = v.currentTime+1/framerate;
    range.value = v.currentTime;
    chart1.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    chart2.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    draw_mice();
};

var reassign_identity_button = document.getElementById('reassign_identity_button');

var video_col = document.getElementById('video_col');

var svg = document.getElementById('svg');

var time_h3 = document.getElementById('time_h3');
var frame_h3 = document.getElementById('frame_h3');
var range_of_re = document.getElementById('rangeOfReassign');
var framerate = 29.95;

var reader = new FileReader();

var json;
var json_stack = [];
var json_stk_top_ptr = 0;
var json_stk_cur_ptr = 0;
var json_stk_btm_ptr = 0;
let json_stk_max = 10;
var can_undo = false;
var can_redo = false;

var btn = $("#play");
var r;
var chart1;
var chart2;
var dataPoints00 = [];
var dataPoints01 = [];
var dataPoints02 = [];
var dataPoints03 = [];
var dataPoints10 = [];
var dataPoints11 = [];
var dataPoints12 = [];
var dataPoints13 = [];
var repeater;
var inpoint=-1;
var outpoint=-1;
var first_load=true;
var frame;
var stripLineSwitch = false;

function log_update(str) {
    var log_text = $("#log_text").append('['+Date().toString().substring(16,24)+'] '+str+"\n");
    log_text.scrollTop(log_text[0].scrollHeight - log_text.height());
}


function compare(p){
    return function(m,n){
        var a = m[p];
        var b = n[p];
        return a - b; //Ascend
    }
}

function dataLeftCompleting(bits, identifier, value){
    value = Array(bits + 1).join(identifier) + value;
    return value.slice(-bits);
}

function toggleDataSeries(e) {
    e.dataSeries.visible = !(typeof (e.dataSeries.visible) === "undefined" || e.dataSeries.visible);
    chart1.render();
    chart2.render();
}

function loadBoxPoints(chart,idx) {
    dataPoints00 = [];
    dataPoints01 = [];
    dataPoints02 = [];
    dataPoints03 = [];
    dataPoints10 = [];
    dataPoints11 = [];
    dataPoints12 = [];
    dataPoints13 = [];

    var framelen = 0;

    for(var i in json[Object.keys(json)[0]]){
        framelen++;
    }

    var tmpChart = new CanvasJS.Chart("miceChart"+chart,{
        zoomEnabled: true,
        axisX:{
            valueFormatString: "####",
            maximum: v.duration*framerate,
            stripLines:[
                {
                    value: Math.round(v.currentTime*framerate),
                    label: ""
                },{
                    startValue:inpoint,
                    endValue:outpoint,
                    color:"#f8ff99"
                }
            ]
        },
        legend:{
            cursor:"pointer",
            itemclick : toggleDataSeries
        },
        toolTip:{
            content:"Frame:{x};Value:{y}",
        },
        data: [{
            type: "line",
            showInLegend: true,
            name:"Corner-point 0",
            dataPoints : dataPoints00,
        },{
            type: "line",
            showInLegend: true,
            name:"Corner-point 1",
            dataPoints : dataPoints01,
        },{
            type: "line",
            showInLegend: true,
            name:"Corner-point 2",
            dataPoints : dataPoints02,
        },{
            type: "line",
            showInLegend: true,
            name:"Corner-point 3",
            dataPoints : dataPoints03,
        }]
    });
    for(let key in json){
        var j;
        for(j=0;j<framelen;++j){
            if(json[key][j].idx===idx) break;
        }
        if(j!==2){//2 means no correct idx
            dataPoints00.push({
                y:json[key][j]["box"][0],
                x:key.split("_",2)[1]
            });
            dataPoints01.push({
                y:json[key][j]["box"][1],
                x:key.split("_",2)[1]
            });
            dataPoints02.push({
                y:json[key][j]["box"][2],
                x:key.split("_",2)[1]
            });
            dataPoints03.push({
                y:json[key][j]["box"][3],
                x:key.split("_",2)[1]
            });
        }
    }
    dataPoints00.sort(compare("x"));
    dataPoints01.sort(compare("x"));
    dataPoints02.sort(compare("x"));
    dataPoints03.sort(compare("x"));
    if(chart===1){
        chart1 = tmpChart;
        chart1.render();
    }else if(chart===2){
        chart2 = tmpChart;
        chart2.render();
    }

}

function loadScore(chart,idx) {
    dataPoints00 = [];
    dataPoints01 = [];
    dataPoints02 = [];
    dataPoints03 = [];
    dataPoints10 = [];
    dataPoints11 = [];
    dataPoints12 = [];
    dataPoints13 = [];

    var framelen = 0;

    for(var i in json[Object.keys(json)[0]]){
        framelen++;
    }
    var tmpChart = new CanvasJS.Chart("miceChart"+chart,{
        zoomEnabled: true,
        axisX:{
            valueFormatString: "####",
            maximum: v.duration*framerate,
            stripLines:[
                {
                    value: Math.round(v.currentTime*framerate),
                    label: ""
                },{
                    startValue:inpoint,
                    endValue:outpoint,
                    color:"#f8ff99"
                }
            ]
        },
        legend:{
            cursor:"pointer",
            itemclick : toggleDataSeries
        },
        toolTip:{
            content:"Frame:{x};Value:{y}",
        },
        data: [{
            type: "line",
            showInLegend: true,
            name:"Score",
            dataPoints : dataPoints00,
        }]
    });
    for(let key in json){
        var j;
        for(j=0;j<framelen;++j){
            if(json[key][j].idx===idx) break;
        }
        if(j!==2){//2 means no correct idx
            dataPoints00.push({
                y:json[key][j]["scores"],
                x:key.split("_",2)[1]
            });
        }
    }
    dataPoints00.sort(compare("x"));
    if(chart===1){
        chart1 = tmpChart;
        chart1.render();
    }else if(chart===2){
        chart2 = tmpChart;
        chart2.render();
    }

}

function loadPoint(chart,n,idx) {
    dataPoints00 = [];
    dataPoints01 = [];
    dataPoints02 = [];
    dataPoints03 = [];
    dataPoints10 = [];
    dataPoints11 = [];
    dataPoints12 = [];
    dataPoints13 = [];


    var framelen = 0;

    for(var i in json[Object.keys(json)[0]]){
        framelen++;
    }
    var tmpChart = new CanvasJS.Chart("miceChart"+chart,{
        zoomEnabled: true,
        axisX:{
            valueFormatString: "####",
            maximum: v.duration*framerate,
            stripLines:[
                {
                    value: Math.round(v.currentTime*framerate),
                    label: ""
                },{
                    startValue:inpoint,
                    endValue:outpoint,
                    color:"#f8ff99"
                }
            ]
        },
        legend:{
            cursor:"pointer",
            itemclick : toggleDataSeries
        },
        toolTip:{
            content:"Frame:{x};Value:{y}",
        },
        data: [{
            type: "line",
            showInLegend: true,
            name:"Point"+(n+1)+" x",
            dataPoints : dataPoints00,
        },{
            type: "line",
            showInLegend: true,
            name:"Point"+(n+1)+" y",
            dataPoints : dataPoints01,
        },{
            type: "line",
            visible: false,
            showInLegend: true,
            name:"Point"+(n+1)+" score",
            dataPoints : dataPoints02,
        }]
    });
    for(let key in json){
        var j;
        for(j=0;j<framelen;++j){
            if(json[key][j].idx===idx) break;
        }
        if(j!==2){//2 means no correct idx
            dataPoints00.push({
                y:json[key][j]["keypoints"][n*3],
                x:key.split("_",2)[1]
            });
            dataPoints01.push({
                y:json[key][j]["keypoints"][n*3+1],
                x:key.split("_",2)[1]
            });
            dataPoints02.push({
                y:json[key][j]["keypoints"][n*3+2]*1000,
                x:key.split("_",2)[1]
            });
        }
    }
    dataPoints00.sort(compare("x"));
    dataPoints01.sort(compare("x"));
    dataPoints02.sort(compare("x"));
    if(chart===1){
        chart1 = tmpChart;
        chart1.render();
    }else if(chart===2){
        chart2 = tmpChart;
        chart2.render();
    }
}


$("#chart1-ctrl").change(function(){
    chartUpdate(1);
});

$("#chart2-ctrl").change(function(){
    chartUpdate(2);
});

function chartUpdate(chart) {
    var selectVal = $("#chart"+chart+"-ctrl option:selected").val();
    if(json===undefined){
        $('#json_err').modal('show');
        $("#play").toggleClass("paused");
        $("#chart"+chart+"-ctrl").val(5);
    }else {
        switch (selectVal) {
            case '5':
                console.log(selectVal);
                loadBoxPoints(chart,chart);
                break;
            case '0':
                console.log(selectVal);
                loadPoint(chart,0,chart);
                break;
            case '1':
                console.log(selectVal);
                loadPoint(chart,1,chart);
                break;
            case '2':
                console.log(selectVal);
                loadPoint(chart,2,chart);
                break;
            case '3':
                console.log(selectVal);
                loadPoint(chart,3,chart);
                break;
            case '4':
                console.log(selectVal);
                loadScore(chart,chart);
                break;
            default:
                break;
        }
    }
}

var undo_btn = document.getElementById("undo_button");
undo_btn.onclick=function () {
    if(can_undo) {
        json = $.extend(true, {}, json_stack_pop());
        chartUpdate(1);
        chartUpdate(2);
        draw_mice();
        document.getElementById("redo_button").style.color = "white";
        can_redo = true;
        log_update("Undo a transaction.");
    }
};

var redo_btn = document.getElementById("redo_button");
redo_btn.onclick=function () {
    if(can_redo){
        if(json_stk_cur_ptr===json_stk_max){
            json_stk_cur_ptr = 0;
        }else json_stk_cur_ptr++;
        json = $.extend(true, {}, json_stack[json_stk_cur_ptr]);
        if(json_stk_cur_ptr === json_stk_top_ptr) {
            document.getElementById("redo_button").style.color = "gray";
            can_redo = false;
        }
        chartUpdate(1);
        chartUpdate(2);
        draw_mice();
        can_undo = true;
        document.getElementById("undo_button").style.color = "white";
        log_update("Redo a transaction.");
    }
};
function json_stack_push(tmp_json) {
    if((json_stk_cur_ptr+1)%(json_stk_max+1)===json_stk_btm_ptr){
        json_stk_cur_ptr = (json_stk_cur_ptr+1)%(json_stk_max+1);
        json_stk_btm_ptr = (json_stk_btm_ptr+1)%(json_stk_max+1);
    }else json_stk_cur_ptr++;
    json_stk_top_ptr = json_stk_cur_ptr;
    json_stack[json_stk_top_ptr] = $.extend(true,{},tmp_json);
    document.getElementById("redo_button").style.color = "gray";
    can_redo = false;
    if(json_stk_top_ptr!==json_stk_btm_ptr){
        document.getElementById("undo_button").style.color = "white";
        can_undo = true;
    }
}

function json_stack_pop() {
    if(json_stk_btm_ptr===json_stk_cur_ptr){
        console.error("Empty Stack!");
        return null;
    }else {
        if(json_stk_cur_ptr===0) json_stk_cur_ptr = json_stk_max;
        else json_stk_cur_ptr--;
        var rtn_json = json_stack[json_stk_cur_ptr];
        if(json_stk_cur_ptr===json_stk_btm_ptr){
            can_undo = false;
            document.getElementById("undo_button").style.color = "gray";
        }
        return rtn_json;
    }
}



var range = document.getElementById('range');

var colorlist = [['rgba(255,0,0,0.8)','rgba(201,0,255,0.8)','rgba(255,49,145,0.8)','rgba(255,138,0,0.8)'],
                 ['rgba(0,0,255,0.8)','rgba(0,88,240,0.8)','rgba(0,181,235,0.8)','rgba(0,255,239,0.8)']];

var ele = false;

var offset,transform;


svg.onload = function makeDraggable(evt){
    svg.addEventListener('mousedown',startDrag);
    svg.addEventListener('mousemove',drag);
    svg.addEventListener('mouseup',endDrag);
    svg.addEventListener('mouseleave',endDrag);

    function getMousePosition(evt){
        var CTM = svg.getScreenCTM();
        return {
            x:(evt.clientX-CTM.e)/CTM.a,
            y:(evt.clientY-CTM.f)/CTM.d
        }
    }

    function startDrag(evt) {
        console.log('startDrag');
        if (evt.target.classList.contains('draggable')) {
            ele = evt.target;
            offset = getMousePosition(evt);
            console.log(offset);

            // Get all the transforms currently on this element
            var transforms = ele.transform.baseVal;

            // Ensure the first transform is a translate transform
            if (transforms.length === 0 ||
                transforms.getItem(0).type !== SVGTransform.SVG_TRANSFORM_TRANSLATE) {
                // Create an transform that translates by (0, 0)
                var translate = svg.createSVGTransform();
                translate.setTranslate(0, 0);

                // Add the translation to the front of the transforms list
                ele.transform.baseVal.insertItemBefore(translate, 0);
            }

            // Get initial translation amount
            transform = transforms.getItem(0);
            offset.x -= transform.matrix.e;
            offset.y -= transform.matrix.f;
        }
    }

    function drag(evt) {
        if (ele) {
            evt.preventDefault();
            var coord = getMousePosition(evt);
            // console.log(coord);
            transform.setTranslate(coord.x - offset.x, coord.y - offset.y);

        }
    }

    function endDrag(evt){

        id = parseFloat(ele.getAttributeNS(null, 'id'));
        id_string = ele.getAttributeNS(null, 'id').toString();
        console.log(id);

        cx = parseFloat(ele.getAttributeNS(null, 'cx'));
        cy = parseFloat(ele.getAttributeNS(null, 'cy'));

        transform = ele.transform.baseVal.getItem(0);
        cx += transform.matrix.e;
        cy += transform.matrix.f;

        json['frame_'+frame][idx_array[id_string[1]]]['keypoints'][id_string[0]] = cx/r;
        json['frame_'+frame][idx_array[id_string[1]]]['keypoints'][Number(id_string[0])+1] = cy/r;
        ele = null;
        json_stack_push(json);
        draw_mice();
        if(id%10===1) chartUpdate(1);
            else chartUpdate(2);
        log_update("Move Identity# "+id%10+",Point#"+(id-id%10)/30+" to "+(cx/r).toFixed(2)+", "+(cy/r).toFixed(2));
    }
};

var id_string, idx_array;

function draw_one_line(idx,pose,sx,sy,ex,ey){

    var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');

    line.setAttributeNS(null, 'x1',  pose[sx]*r);
    line.setAttributeNS(null, 'x2',  pose[ex]*r);
    line.setAttributeNS(null, 'y1',  pose[sy]*r);
    line.setAttributeNS(null, 'y2',  pose[ey]*r);

    line.setAttributeNS(null, 'stroke', colorlist[idx-1][0]);
    line.setAttributeNS(null, 'stroke-width', 2);

    svg.appendChild(line);

}

function draw_one_point(idx,pose,x,y,id){

    var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');

    circle.setAttributeNS(null, 'cx', pose[x]*r);
    circle.setAttributeNS(null, 'cy', pose[y]*r);
    circle.setAttributeNS(null, 'r',  5);
    //console.log(x+"_"+idx);
    circle.setAttributeNS(null, 'fill', colorlist[idx-1][x/3]);
    circle.setAttributeNS(null, 'class', 'draggable');
    circle.setAttributeNS(null, 'id',id.toString()+idx.toString());
    svg.appendChild(circle);
}


function draw_one_mouse(idx,pose){
    draw_one_line(idx,pose,0,1,3,4);
    draw_one_line(idx,pose,0,1,6,7);
    draw_one_line(idx,pose,0,1,9,10);
    draw_one_point(idx,pose,0,1,'0');
    draw_one_point(idx,pose,3,4,'3');
    draw_one_point(idx,pose,6,7,'6');
    draw_one_point(idx,pose,9,10,'9');

    var text = document.createElementNS('http://www.w3.org/2000/svg','text');
    text.setAttributeNS(null,'x', (pose[0]-20)*r);
    text.setAttributeNS(null,'y', (pose[1]-20)*r);
    text.setAttributeNS(null,'font-size',20);
    text.setAttributeNS(null,'fill', colorlist[idx-1][0]);
    text.textContent = idx;
    svg.appendChild(text);
}


function draw_mice(){
    frame = Math.round(v.currentTime * framerate);
    p = v.currentTime /v.duration;
    range.value = v.currentTime;
    time_h3.innerHTML = dataLeftCompleting(2,'0',Math.floor(v.currentTime/3600)) + ':'
        + dataLeftCompleting(2,'0',Math.floor(v.currentTime/60%60)) + ':'
        + dataLeftCompleting(2,'0',Math.floor(v.currentTime%60));
    frame_h3.innerHTML =  String(Math.round(v.currentTime*framerate));
    let mice = json['frame_' + frame];
    idx_array = [0,0,0];
    while (svg.lastChild) {
        svg.removeChild(svg.lastChild);
    }
    if(undefined!==mice){
        for (i = 0; i<mice.length; i++){
            pose = mice[i].keypoints;
            if(pose!==undefined) draw_one_mouse(mice[i].idx,pose);
            idx_array[mice[i].idx] = i;
        }
    }
}
//Strip lines update
function stripLineUpdate(){
    setTimeout(function () {
        if(stripLineSwitch){
            chart1.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
            chart2.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
        }//Update every 0.5s
    },1000)
}


v.addEventListener('play', function() {
    repeater = window.setInterval(function() {

        if(v.ended){
            btn.toggleClass("paused");
            clearInterval(i)
        }
        draw_mice();
        stripLineUpdate();

    }, 1000/framerate);

}, false);
v.addEventListener('pause', function() {
    clearInterval(repeater);
    chart1.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    chart2.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
}, false);

$(document).ready(function() {
    btn.click(function() {
        btn.toggleClass("paused");
        return false;
    });
});


play_button.onclick = function(){
    if(json===undefined){
        $('#json_err').modal('show');
        $("#play").toggleClass("paused");
    }else {
        if (v.paused) {
            v.play();
        } else {
            v.pause();
        }
    }

};
var tmp_blob;
var save_button = document.getElementById('save_button')
save_button.onclick = function(){
    var eleLink = document.createElement('a');
    eleLink.download = 'output.json';
    eleLink.style.display = 'none';
    tmp_blob = new Blob([JSON.stringify(json)]);
    eleLink.href = URL.createObjectURL(tmp_blob);
    document.body.appendChild(eleLink);
    eleLink.click();
    document.body.removeChild(eleLink);
};
var tmp_str_json;
var curate_button = document.getElementById('curate_button');
curate_button.onclick = function(){
    //$('#ini_modal').modal('show');
    tmp_str_json = JSON.stringify(json);
    pyodide.runPython("args.frame_start="+inpoint);
    pyodide.runPython("args.frame_end="+outpoint);
    pyodide.runPython("args.num_pose=4");
    pyodide.runPython("args.max_pid_id_setting=2");
    pyodide.runPython("import js");
    pyodide.runPython("str_json = js.tmp_str_json");
    pyodide.runPython("track_forJson = json.loads(str_json)");
    pyodide.runPython("post_process_tracking(track_forJson,args)");
    pyodide.runPython("str_json = json.dumps(track_forJson)");
    tmp_str_json = pyodide.pyimport('str_json');
    json = JSON.parse(tmp_str_json);
    json_stack_push(json);
    chartUpdate(1);
    chartUpdate(2);
    $('#ini_modal').modal('hide');
    log_update("Curate Finished!");
    alert("Curate Finished!");
};

range.addEventListener('input', function () {
    v.currentTime = range.value;
    frame_h3.textContent =  String(Math.round(v.currentTime*framerate));
    chart1.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    chart2.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    draw_mice();
});
/*
frame_h3.addEventListener('change',function () {
    time_h3.textContent = parseInt(frame_h3.innerHTML)/framerate > v.duration? v.duration : frame_h3.innerHTML/framerate;
    v.currentTime = parseInt(frame_h3.innerHTML)/framerate > v.duration? v.duration : frame_h3.innerHTML/framerate;
    //frame_h3.textContent =  String(Math.round(v.currentTime*framerate));
    chart1.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    chart2.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    draw_mice();
});*/


var new_identity_input = document.getElementById("new_identity_input");
var old_identity_input = document.getElementById("old_identity_input");


reassign_identity_button.onclick = function(){
    /*if(new_identity_input.value==='none' || old_identity_input.value==='none'){
        alert("Please specify two identities!");
    }else*/
     if(inpoint===-1 || outpoint===-1){
        alert("Please specify the Range!");
    } else {
        //var new_identity = Number(new_identity_input.value);
        //var old_identity = Number(old_identity_input.value);
        var new_identity = 1;
        var old_identity = 2;
        for(var tmp = inpoint;tmp<=outpoint;++tmp){
            var tmp_frame = json['frame_'+tmp];
            var tmp_idx_array = [0,0,0];
            if(tmp_frame!==undefined){
                if (tmp_frame[0].idx !== undefined && tmp_frame[1].idx!==undefined){
                    for (i = 0; i<tmp_frame.length; i++){
                        tmp_idx_array[tmp_frame[i].idx] = i;
                    }
                    json['frame_'+tmp][tmp_idx_array[old_identity]]['idx'] = new_identity;
                    json['frame_'+tmp][tmp_idx_array[new_identity]]['idx'] = old_identity;
                }else if (tmp_frame[0].idx === undefined && tmp_frame[1].idx!==undefined){
                    if(tmp_frame[1].idx===1)tmp_frame[1].idx=2;else tmp_frame[1].idx=1;
                    tmp_frame[0] = tmp_frame[1];
                    tmp_frame[1] = {};

                }else if (tmp_frame[0].idx !== undefined && tmp_frame[1].idx===undefined){
                    if(tmp_frame[0].idx===1)tmp_frame[0].idx=2;else tmp_frame[0].idx=1;
                    tmp_frame[1] = tmp_frame[0];
                    tmp_frame[0] = {};
                }

            }

        }
        json_stack_push(json);
        draw_mice();
        chartUpdate(1);
        chartUpdate(2);
        log_update("Identities reassigned From Frame "+inpoint+" to "+outpoint);
    }
};


var speed_ctrl = document.getElementById('speed-control');
speed_ctrl.addEventListener('change',function () {
    v.playbackRate=speed_ctrl.value;
});

function check() {
    var video_file = document.getElementById("video_input");
    var json_input = document.getElementById('json_input');

    if(video_file.files[0]===undefined || json_input.files[0]===undefined){
        alert("Please import all required files!");
    }else {
        document.getElementById('video').src= URL.createObjectURL(video_file.files[0]);
        svg.setAttribute("viewBox", "0 0 "+v.offsetWidth + " " +v.offsetHeight);
        reader.readAsText(json_input.files[0]);
        reader.onload = function(){
            json = JSON.parse(this.result);
            json_stack_push(json);
            //Demo Code- Load json data into CanvasJS Chart
            draw_mice();
        };
        var tmp_int;
        range.value = 0;
        tmp_int=setInterval(function () {
            if(!isNaN(v.duration)){
                console.log(v.duration);
                range.max = v.duration;

                if(v.offsetWidth/v.offsetHeight > v.videoWidth/v.videoHeight){
                    r = v.offsetHeight/v.videoHeight;
                    svg.setAttribute("viewBox", (v.videoWidth*v.offsetHeight/v.videoHeight-v.offsetWidth)/2 + " 0 "+v.offsetWidth + " " +v.offsetHeight);
                }else{
                    r = v.offsetWidth/v.videoWidth;
                    svg.setAttribute("viewBox", "0 "+(v.videoHeight*v.offsetWidth/v.videoWidth-v.offsetHeight)/2 +' '+ v.offsetWidth + " " +v.offsetHeight);
                }

                loadBoxPoints(1,1);
                loadBoxPoints(2,2);
                clearInterval(tmp_int);
            }
        },400);
        framerate = Number(document.getElementById("framerate_input").value);

        if(first_load){
            //temp block python
                languagePluginLoader.then(function () {
                    console.log(pyodide.runPython('import sys\nsys.version'));
                    var tmp_str = document.getElementById("py_script").innerHTML;
                    $('#input_modal').modal('hide');
                    $('#ini_modal').modal('show');
                    pyodide.loadPackage("scipy").then(() => {
                        try{
                            pyodide.runPython(tmp_str);
                            $('#ini_modal').modal('hide');
                        }catch (e) {
                            $('#ini_modal').modal('hide');
                            console.log(String(e));
                            document.getElementById("py_err_text").innerHTML  = String(e);
                            $('#py_err').modal('show');
                        }
                    });
                });
            first_load = false;
        }else {
            $('#input_modal').modal('hide');
        }


    }
}

function show_filename(id,fakepath) {
    var tmp_str = fakepath.substring(fakepath.lastIndexOf("\\")+1);
    document.getElementById(id).innerHTML = '<a href="#" data-toggle="popover" data-content="'+tmp_str+
    '">'+tmp_str.substring(0, Math.min(tmp_str.length, 15))+'...'+'</a>';
    $(document).ready(function(){
        $('[data-toggle="popover"]').popover();
    });
}
