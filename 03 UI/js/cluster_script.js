var v = document.getElementById('video');

var play_button = document.getElementById('play');

var back_button = document.getElementById('back_button');
back_button.onclick = function(){
    v.currentTime = v.currentTime-1/framerate;
    range.value = v.currentTime;
    chart_clip.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
};

var forward_button = document.getElementById('forward_button');
forward_button.onclick = function(){
    v.currentTime = v.currentTime+1/framerate;
    range.value = v.currentTime;
    chart_clip.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
};


//var video_col = document.getElementById('video_col');
var framerate = 30;

var svg = document.getElementById('svg');

var time_h3 = document.getElementById('time_h3');
var frame_h3 = document.getElementById('frame_h3');
var clip_text = document.getElementById('current_clip');

var reader = new FileReader();

var json_clip;//JSON of clips info
var json_tree;//JSON of dendrogram
var json_tree_max_level = 0;

var node_counter = 0;

var clip_set = new Set();// A set of all clips inorder traversal id

var video_array=[];//Array of n videos
var video_src_array=[];//
var video_name_array=[];
var video_framerate_array=[];
var data_array=[];//Scatter points' info, n array(s), content: {x: y: color:}
var data_array_disp=[];//for plot display

var range = document.getElementById('range');

var btn = $("#play");

function log_update(str) {
    var log_text = $("#log_text").append('['+Date().toString().substring(16,24)+'] '+str+"\n");
    log_text.scrollTop(log_text[0].scrollHeight - log_text.height());
}

function timestamp_update() {
    time_h3.innerHTML = dataLeftCompleting(2,'0',Math.floor(v.currentTime/3600)) + ':'
        + dataLeftCompleting(2,'0',Math.floor(v.currentTime/60%60)) + ':'
        + dataLeftCompleting(2,'0',Math.floor(v.currentTime%60));
    frame_h3.textContent =  String(Math.round(v.currentTime*framerate));
    if(frame_clip_map.get(Math.round(v.currentTime*framerate))!==undefined){
        clip_text.textContent = frame_clip_map.get(Math.round(v.currentTime*framerate));
    }else{
        clip_text.textContent = "---";
    }
}

var chart_clip;//Scatter Plot
function chart_update(data) {
    var tmp_value = 0;
    if(video_sel.value!==undefined) tmp_value = video_sel.value;
    chart_clip = new CanvasJS.Chart("chartContainer", {
        animationEnabled: true,
        animationDuration: 300,
        zoomEnabled: true,
        axisX: {
            title:"Frame#",
            minimum:0,
            maximum: v.duration*framerate,
            includeZero: true,
            stripLines:[
                {
                    value: 0,
                    label: ""
                }
            ]
        },
        axisY:{
            title: "Cluster#",
            includeZero: false
        },
        data: [{
            type: "scatter",
            toolTipContent: "Cluster:# {y}",
            name: "Cluster 1",
            dataPoints: data[tmp_value],
        }]
    });
    chart_clip.render();
}


var nodeDataArray = [];
var repeater;

//Related of Map
//A map of every node's contained frames
//Ini when json_clip and json_tree reading in
//Update when merge or delete or move
var node_frame_map = new Map();
var frame_clip_map = new Map();
function clip_map_insert(id,frame) {
    frame_clip_map.set(frame,id);
    if(node_frame_map.get(id)===undefined){
        node_frame_map.set(id,[frame]);
    }else {
        node_frame_map.get(id).push(frame);
    }
}
/*function cluster_map_insert(id,child1,child2) {
    node_frame_map.set(id,node_frame_map.get(child1).concat(node_frame_map.get(child2)));
}*/
function highlight_scatter(id) {
    data_array_disp = JSON.parse(JSON.stringify(data_array));
    function f(node_id) {
        var tmp_array2 = [];
        var tmp_node = myDiagram.findNodeForKey(node_id);
        if(tmp_node.isTreeLeaf){
            tmp_array2 = tmp_node.data.frames;
        }else{
            var it = tmp_node.findTreeChildrenNodes();
            while (it.next()){
                tmp_array2 = tmp_array2.concat(f(it.value.data.key));
            }
        }
        return tmp_array2;
    }
    var tmp_array = f(id);
    for(var i=0;i<data_array_disp[video_sel.value].length;++i){
        if(tmp_array.indexOf(data_array_disp[video_sel.value][i].x)!== -1){
            data_array_disp[video_sel.value][i].color = "#25f8ff";
        }
    }
    chart_update(data_array_disp);
    range.value = v.currentTime = Math.min(...tmp_array)/framerate;
    timestamp_update();
    chart_clip.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
}
//End of Map related function

function dataLeftCompleting(bits, identifier, value){
    value = Array(bits + 1).join(identifier) + value;
    return value.slice(-bits);
}

function color_generate(max, value) {
    var ratio = value/max;
    var color_str = "000000";
    if(ratio >= 0 && ratio < 0.5){
        color_str = "#" + dataLeftCompleting(2,'0',Math.ceil(ratio/0.5*225+30).toString(16).toUpperCase())
            + dataLeftCompleting(2,'0',Math.ceil(ratio/0.5*92+150).toString(16).toUpperCase()) + "00";
    }else if(ratio >= 0.5 && ratio <= 1){
        color_str = "#FF"+dataLeftCompleting(2,'0',Math.ceil((1-ratio)/0.5*242).toString(16).toUpperCase())+"00";
    }
    return color_str;
}

function getMP (e) {
    var e = e || window.event;
    return {
        x : e.pageX || e.clientX + (document.documentElement.scrollLeft || document.body.scrollLeft),
        y : e.pageY || e.clientY + (document.documentElement.scrollTop || document.body.scrollTop)
    }
}

v.addEventListener('play', function() {
    repeater = window.setInterval(function() {
        range.value = v.currentTime;
        timestamp_update();
        if(v.ended){
            clearInterval(repeater);
            btn.toggleClass("paused");
        }
        chart_clip.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
    }, 1000/29.85);
}, false);
v.addEventListener('pause', function() {
    clearInterval(repeater);
}, false);

$(document).ready(function() {
    btn.click(function() {
        btn.toggleClass("paused");
        return false;
    });
});


play_button.onclick = function(){
    if (v.paused) {
        v.play();
    } else {
        v.pause();
    }

};


range.addEventListener('input', updateValue);
function updateValue(e) {
    v.currentTime = range.value;
    timestamp_update();
    chart_clip.axisX[0].stripLines[0].set("value",Math.round(v.currentTime*framerate));
}

function check_undo_n_redo(e) {
    if(myDiagram.model.undoManager.canRedo()){
        document.getElementById("redo_button").style.color = "white";
    }else{
        document.getElementById("redo_button").style.color = "gray";
    }
    if(myDiagram.model.undoManager.canUndo()){
        document.getElementById("undo_button").style.color = "white";
    }else{
        document.getElementById("undo_button").style.color = "gray";
    }
}

var json_input = document.getElementById('json_input_clip');
json_input.addEventListener('change',function(){
    reader.readAsText(json_input.files[0]);
    reader.onload = function(){
        json_clip = JSON.parse(this.result);
        for(var i=0;i<json_clip.length;++i){
            clip_set.add(json_clip[i]["inorder_traversal_id"]);//Set of all clips traversal ID
        }
        var tmp_max = Math.max(...Array.from(clip_set));
        for(var i=0;i<json_clip.length;++i){
            var tmp_data_array=[];

            //console.log(json_clip[i]["video_path"]);
            if(video_array.indexOf(json_clip[i]["video_path"]) === -1){
                video_array.push(json_clip[i]["video_path"]);//Find a new video path
                for(var j=0;j<json_clip[i]["frame_keys"].length;++j){
                    tmp_data_array.push({
                        x:Number(json_clip[i]["frame_keys"][j].split('_')[1]),
                        y:Number(json_clip[i]["inorder_traversal_id"]),
                        color: color_generate(tmp_max,json_clip[i]["inorder_traversal_id"])
                    });
                    clip_map_insert(Number(json_clip[i]["inorder_traversal_id"]),Number(json_clip[i]["frame_keys"][j].split('_')[1]));
                }
                data_array.push(tmp_data_array);//Same index with video
            }else if(video_array.indexOf(json_clip[i]["video_path"]) !== -1){
                for(var j=0;j<json_clip[i]["frame_keys"].length;++j){
                    data_array[video_array.indexOf(json_clip[i]["video_path"])].push({
                        x:Number(json_clip[i]["frame_keys"][j].split('_')[1]),
                        y:Number(json_clip[i]["inorder_traversal_id"]),
                        color: color_generate(tmp_max,json_clip[i]["inorder_traversal_id"])
                    });
                    clip_map_insert(Number(json_clip[i]["inorder_traversal_id"]),Number(json_clip[i]["frame_keys"][j].split('_')[1]));
                }

            }

        }
        //data_array_disp = JSON.parse(JSON.stringify(data_array));//Deep copy of data_array
        var innerStr="";
        for(i=0;i<video_array.length;++i){
            var s = video_array[i];
            innerStr=innerStr+'<tr> <td><a href="#" data-toggle="popover" data-content="'+s+
               '">'+s.substring(0, Math.min(s.length, 10))+'...'+'</a></td>'+
                '<td><input type="text" class="form-control" id="video_name_'+i+'" value="video_'+i+
                '"></td><td><input type="text" class="form-control" id="video_framerate_'+i+'" value="30"></td>' +
                '<td><button class="btn btn-secondary" style="width: 80px;height: 35px;margin: 5px;text-align: center;padding: 5px 7px;">'+
               'Import<input style="top: -35px" type="file" class="custom-file-input" id="input_video_'+i+'" aria-describedby="inputGroupFileAddon01"'+
                ' onchange="video_import('+i+',\'input_video_'+i+'\',\'video_name_'+i+'\')"> </button></td></tr>'

        }

        document.getElementById("video_table").innerHTML=innerStr;

        $(document).ready(function(){
            $('[data-toggle="popover"]').popover();
        });

    };
});
var inspector;
var selected_node;
var json_input_tree = document.getElementById('json_input_tree');
json_input_tree.addEventListener("change", function () {
    reader.readAsText(json_input_tree.files[0]);
    reader.onload = function() {
        json_tree = JSON.parse(this.result);
        // create a tree if not load from GOjs
        if(json_tree.class !== "go.TreeModel"){
            node_counter=json_clip.length;
            for(var i=0;i<json_tree.length;++i){
                var tmp_data;
                if(json_tree[i][0]<json_clip.length){
                    tmp_data={key: json_tree[i][0],parent: node_counter,name:'Clip'+json_tree[i][0],frames:node_frame_map.get(json_tree[i][0])};
                    nodeDataArray.push(tmp_data);
                }else {
                    tmp_data={key: json_tree[i][0],parent: node_counter,name:'Cluster'+json_tree[i][0]};
                    nodeDataArray.push(tmp_data);
                }
                if(json_tree[i][1]<json_clip.length){
                    tmp_data={key: json_tree[i][1],parent: node_counter,name:'Clip'+json_tree[i][1],frames:node_frame_map.get(json_tree[i][1])};
                    nodeDataArray.push(tmp_data);
                }else {
                    tmp_data={key: json_tree[i][1],parent: node_counter,name:'Cluster'+json_tree[i][1]};
                    nodeDataArray.push(tmp_data);
                }
                //cluster_map_insert(counter,json_tree[i][0],json_tree[i][1]);
                node_counter++;
            }
        }

        //tree
        var $ = go.GraphObject.make;  // for conciseness in defining templates

        myDiagram =
            $(go.Diagram, "myDiagramDiv",
                {
                    "undoManager.isEnabled": true,
                    allowMove: false,
                    allowCopy: false,
                    allowDelete: true,
                    allowHorizontalScroll: false,
                    layout:
                        $(go.TreeLayout,
                            {
                                alignment: go.TreeLayout.AlignmentStart,
                                angle: 0,
                                compaction: go.TreeLayout.CompactionNone,
                                layerSpacing: 16,
                                layerSpacingParentOverlap: 1,
                                nodeIndentPastParent: 1.0,
                                nodeSpacing: 0,
                                setsPortSpot: false,
                                setsChildPortSpot: false
                            })
                });

        var cxElement = document.getElementById("contextMenu");

        var myContextMenu = $(go.HTMLInfo, {
            show: showContextMenu,
            hide: hideContextMenu
        });

        myDiagram.nodeTemplate =
            $(go.Node,
                {
                    selectionAdorned: false,
                    doubleClick: function(e, node) {
                        var cmd = myDiagram.commandHandler;
                        console.log(node.key);
                        /*
                        if (node.isTreeExpanded) {
                            if (!cmd.canCollapseTree(node)) return;
                        } else {
                            if (!cmd.canExpandTree(node)) return;
                        }*/
                        e.handled = true;
                        highlight_scatter(node.key);
                        /*
                        if (node.isTreeExpanded) {
                            cmd.collapseTree(node);
                        } else {
                            cmd.expandTree(node);
                        }*/
                    }
                },{contextMenu: myContextMenu},
                $("TreeExpanderButton",
                    {
                        "ButtonBorder.fill": "whitesmoke",
                        "ButtonBorder.stroke": null,
                        "_buttonFillOver": "rgba(0,128,255,0.25)",
                        "_buttonStrokeOver": null
                    }),
                $(go.Panel, "Horizontal",
                    { position: new go.Point(18, 0) },
                    new go.Binding("background", "isSelected", function(s) { return (s ? "lightblue" : "white"); }).ofObject(),
                    $(go.Picture,
                        {
                            width: 18, height: 18,
                            margin: new go.Margin(0, 4, 0, 0),
                            imageStretch: go.GraphObject.Uniform
                        },

                        new go.Binding("source", "isTreeExpanded", imageConverter).ofObject(),
                        new go.Binding("source", "isTreeLeaf", imageConverter).ofObject()),
                    $(go.TextBlock,
                        {
                            font: '9pt Verdana, sans-serif'},
                        new go.Binding("text", "name", function(s) { return  s; }))
                )  // end Horizontal Panel
            );  // end Node


        // without lines
        myDiagram.linkTemplate = $(go.Link);
        if(json_tree.class === "go.TreeModel"){
            myDiagram.model = go.Model.fromJSON(this.result);
        }else {
            myDiagram.model = new go.TreeModel(nodeDataArray);
        }
        //Get max level of tree
        var diagram_roots = myDiagram.findTreeRoots();
        while(diagram_roots.next()){
            var tmp_set = diagram_roots.value.findTreeParts();
            var tmp_it = tmp_set.iterator;
            while(tmp_it.next()){
                var tmp_level = myDiagram.findNodeForKey(tmp_it.value.data.key).findTreeLevel();
                json_tree_max_level = json_tree_max_level < tmp_level ? tmp_level : json_tree_max_level;
            }
        }
        //Set Level selection
        var level_innerOpt="";
        for(i = 0;i<=json_tree_max_level;++i){
            if(i===json_tree_max_level){
                level_innerOpt=level_innerOpt+'<option value='+i+' selected>Level'+ i +'</option>';
            }else{
                level_innerOpt=level_innerOpt+'<option value="'+i+'">Level'+ i +'</option>';
            }
        }
        document.getElementById("level-select").innerHTML=level_innerOpt;


        myDiagram.contextMenu = myContextMenu;
        myDiagram.addModelChangedListener(function (e) {
            if(myDiagram.model.undoManager.canRedo()){
                document.getElementById("redo_button").style.color = "white";
            }else{
                document.getElementById("redo_button").style.color = "gray";
            }
            if(myDiagram.model.undoManager.canUndo()){
                document.getElementById("undo_button").style.color = "white";
            }else{
                document.getElementById("undo_button").style.color = "gray";
            }
        });
        //myDiagram.model.addChangedListener();

        cxElement.addEventListener("contextmenu", function(e) {
            e.preventDefault();
            return false;
        }, false);
        function hideCX() {
            if (myDiagram.currentTool instanceof go.ContextMenuTool) {
                myDiagram.currentTool.doCancel();
            }
        }
        function showContextMenu(obj, diagram, tool) {
            // Show only the relevant buttons given the current state.
            var cmd = diagram.commandHandler;
            var hasMenuItem = false;
            function maybeShowItem(elt, pred) {
                if (pred) {
                    elt.style.display = "block";
                    hasMenuItem = true;
                } else {
                    elt.style.display = "none";
                }
            }

            maybeShowItem(document.getElementById("delete"), cmd.canDeleteSelection());

            // Now show the whole context menu element
            if (hasMenuItem) {
                cxElement.classList.add("show-menu");
                // we don't bother overriding positionContextMenu, we just do it here:

                cxElement.style.left = getMP().x + 5 + "px";
                cxElement.style.top = getMP().y + "px";
            }

            // Optional: Use a `window` click listener with event capture to
            //           remove the context menu if the user clicks elsewhere on the page
            window.addEventListener("click", hideCX, true);
        }
        function hideContextMenu() {
            cxElement.classList.remove("show-menu");
            // Optional: Use a `window` click listener with event capture to
            //           remove the context menu if the user clicks elsewhere on the page
            window.removeEventListener("click", hideCX, true);
            }

        myDiagram.select(myDiagram.nodes.first());
        inspector = new Inspector('myInspectorDiv', myDiagram,
            {
                // allows for multiple nodes to be inspected at once
                multipleSelection: true,
                // max number of node properties will be shown when multiple selection is true
                showSize: 4,
                // when multipleSelection is true, when showAllProperties is true it takes the union of properties
                // otherwise it takes the intersection of properties
                //showAllProperties: true,
                // uncomment this line to only inspect the named properties below instead of all properties on each object:
                // includesOwnProperties: false,
                properties: {

                    // key would be automatically added for nodes, but we want to declare it read-only also:
                    "key": { readOnly: true, show: Inspector.showIfPresent },
                    // color would be automatically added for nodes, but we want to declare it a color also:
                    "parent": { show: Inspector.showIfPresent },
                    // Comments and LinkComments are not in any node or link data (yet), so we add them here:
                    "names": { show: Inspector.showIfPresent },
                    "frames": { show: false },
                }
            });
    };

});
function cxcommand(event, val) {
    if (val === undefined) val = event.currentTarget.id;
    var diagram = myDiagram;
    selected_node = myDiagram.selection.iterator;
    selected_node.next();
    selected_node = selected_node.value;
    switch (val) {
        case "delete":
            log_update("Delete " + selected_node.data.name);
            diagram.commandHandler.deleteSelection();
            break;
        case "delete_subtree":
            log_update("Delete " + selected_node.data.name + " and its subtree");
            myDiagram.removeParts(selected_node.findTreeParts());
            break;
        case "rename":
            document.getElementById("ModalLabel5").innerText = "Rename "+selected_node.data.name+':';
            document.getElementById("node_name").value = selected_node.data.name;
            $('#rename_modal').modal('show');
            break;
        case "move":
            document.getElementById("ModalLabel4").innerText = "Move "+selected_node.data.name+' to:';
            document.getElementById("move_target_node").value = '';
            document.getElementById("move_target_node").setAttribute("placeholder","Target Cluster #");
            $('#move_modal').modal('show');
    }
    diagram.currentTool.stopTool();
}

var speed_ctrl = document.getElementById('speed-control');
speed_ctrl.addEventListener('change',function () {
    v.playbackRate=speed_ctrl.value;
});

var video_sel = document.getElementById('video-select');
video_sel.addEventListener('change',function () {
    $('#json_ini').modal('show');
    //console.log(video_sel.value);
    document.getElementById('video').src=video_src_array[video_sel.value];
    framerate = video_framerate_array[video_sel.value];
    console.log(chart_clip.data[0].dataPoints.length);
    var tmp_int;
    range.value = 0;
    tmp_int=setInterval(function () {
        if(!isNaN(v.duration)){
            console.log(v.duration);
            range.max = v.duration;
            chart_update(data_array);
            setTimeout(function () {
                $('#json_ini').modal('hide');
            },1000);
            clearInterval(tmp_int);
            console.log(v.duration+"aft");
        }
    },200);
});

var level_sel = document.getElementById('level-select');
level_sel.addEventListener('change',function () {
    var diagram_roots = myDiagram.findTreeRoots();

    while(diagram_roots.next()){
        var tmp_value = parseInt(level_sel.value);
        console.log(tmp_value);
        if(tmp_value === 0){
            diagram_roots.value.collapseTree(1);

        }else if (tmp_value === json_tree_max_level){
            diagram_roots.value.expandTree(tmp_value+1);
        }else{
            diagram_roots.value.expandTree(tmp_value+1);
            diagram_roots.value.collapseTree(tmp_value+1);
        }

    }
});

// takes a property change on either isTreeLeaf or isTreeExpanded and selects the correct image to use
function imageConverter(prop, picture) {
    var node = picture.part;
    if (node.isTreeLeaf) {
        return "../image/document.png";
    } else {
        if (node.isTreeExpanded) {
            return "../image/opened-folder.png";
        } else {
            return "../image/folder.png";
        }
    }
}

function video_import(index,id,video_name) {
    var video_file = document.getElementById(id).files[0];
    var video_url = URL.createObjectURL(video_file);
    console.log(video_url);
    video_src_array[index]=video_url;
    video_name_array[index]=document.getElementById(video_name).value;
    video_framerate_array[index]=document.getElementById('video_framerate_'+index).value;
}
var tmp_interval;
function page_ini() {
    var innerOpt="";
    for(i = 0;i<video_array.length;++i){
        if(i===0){
            innerOpt='<option value="0" selected>'+ video_name_array[i] +'</option>';
        }else{
            innerOpt=innerOpt+'<option value="'+i+'">'+ video_name_array[i] +'</option>'
        }
    }
    document.getElementById("video-select").innerHTML=innerOpt;
    //$('#json_ini').modal('show');
    document.getElementById('video').src=video_src_array[0];
    framerate = video_framerate_array[0];

    range.value = 0;
    tmp_interval=setInterval(function () {
        if(!isNaN(v.duration)){
            console.log(v.duration);
            range.max = v.duration;
            //$('#json_ini').modal('hide');
            //Scatter Plot
            frame_h3.innerHTML =  String(Math.round(v.currentTime*framerate));
            time_h3.innerHTML = dataLeftCompleting(2,'0',Math.floor(v.currentTime/3600)) + ':'
                + dataLeftCompleting(2,'0',Math.floor(v.currentTime/60%60)) + ':'
                + dataLeftCompleting(2,'0',Math.floor(v.currentTime%60));

            chart_update(data_array);
            clearInterval(tmp_interval);
        }
    },400);
}

function operation_checkpoint() {
    var tmp_flag = true;
    for(i = 0;i<video_array.length;++i){
        if(video_src_array[i]===undefined){
            tmp_flag = false;
            break;
        }
    }
    if(!tmp_flag){
        alert("Please import all videos!");
    }else {
        $('#input_modal_video').modal('hide');
        $('#json_input_modal_tree').modal('show');
    }
}

var tmp_blob;
var save_button = document.getElementById('save_button');
save_button.onclick = function(){
    var eleLink = document.createElement('a');
    eleLink.download = 'output.json';
    eleLink.style.display = 'none';
    tmp_blob = new Blob([myDiagram.model.toJson()]);
    eleLink.href = URL.createObjectURL(tmp_blob);
    document.body.appendChild(eleLink);
    eleLink.click();
    document.body.removeChild(eleLink);
};


