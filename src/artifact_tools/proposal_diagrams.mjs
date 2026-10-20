import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from 'file:///C:/Users/李静/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

// Native, editable replacements for the three diagrams in the supplied deck.
const out = 'D:/My_project_li/docs/presentations/proposal_revision_20261015/analysis/diagrams.pptx';
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const blue = '#104495', light = '#83A8EE', ink = '#202020';
function box(slide, name, x, y, w, h, text, size=20, fill='#FFFFFF', stroke=light, bold=false, geometry='roundRect') {
  const s=slide.shapes.add({name,geometry,position:{left:x,top:y,width:w,height:h},fill,
    line:{style:'solid',fill:stroke,width:stroke==='none'?0:1.2}});
  s.text=text;
  s.text.style={typeface:'微软雅黑',fontSize:size,color:ink,bold,alignment:'center',verticalAlignment:'middle'};
  s.text.insets=5;
  return s;
}
function label(slide,name,x,y,w,h,text,size=18,color=ink,bold=false){
  const s=box(slide,name,x,y,w,h,text,size,'none','none',bold,'textbox');
  s.text.style={typeface:'微软雅黑',fontSize:size,color,bold,alignment:'center',verticalAlignment:'middle'};
  return s;
}
function link(slide,a,b,from,to,dashed=false,both=false,color='#146BE3',arrow=true){
  return slide.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,
    line:{style:dashed?'dashed':'solid',fill:color,width:color==='#EF942B'?2.8:1.8},
    head:{type:arrow&&both?'triangle':'none',width:'sm',length:'sm'},tail:{type:arrow?'triangle':'none',width:'sm',length:'sm'}});
}

function primitive(slide,name,x,y,w,h,geometry,fill,stroke=blue,rotation=0){
  return slide.shapes.add({name,geometry,position:{left:x,top:y,width:w,height:h,rotation},fill,
    line:{style:'solid',fill:stroke,width:stroke==='none'?0:1.5}});
}
function drone(slide,id,cx,cy,k=1){
  const anchor=box(slide,`${id}_anchor`,cx-61*k,cy-31*k,122*k,62*k,'',12,'none','none');
  primitive(slide,`${id}_arm1`,cx-43*k,cy-3*k,86*k,6*k,'rect',blue,'none',27);
  primitive(slide,`${id}_arm2`,cx-43*k,cy-3*k,86*k,6*k,'rect',blue,'none',-27);
  for(const [idx,dx,dy] of [[0,-42,-22],[1,42,-22],[2,-42,22],[3,42,22]]){
    primitive(slide,`${id}_rotor${idx}`,cx+(dx-17)*k,cy+(dy-7)*k,34*k,14*k,'ellipse','#F4F8FF',blue);
    primitive(slide,`${id}_motor${idx}`,cx+(dx-3)*k,cy+(dy-3)*k,6*k,6*k,'ellipse',blue,'none');
  }
  primitive(slide,`${id}_body`,cx-10*k,cy-20*k,20*k,40*k,'roundRect',blue,'none');
  primitive(slide,`${id}_canopy`,cx-6*k,cy-14*k,12*k,10*k,'roundRect','#53A8F3','none');
  return anchor;
}
function task(slide,id,x,y,k=1){
  primitive(slide,`${id}_back`,x+7*k,y-6*k,25*k,31*k,'roundRect','#EDF4FF',light);
  const a=primitive(slide,`${id}_front`,x,y,25*k,31*k,'roundRect','#FFF4E7','#F19637');
  for(let j=0;j<3;j++)primitive(slide,`${id}_line${j}`,x+5*k,y+(8+j*6)*k,15*k,1.4*k,'rect','#EF9C48','none');
  return a;
}
function chip(slide,id,cx,cy,k=1){
  const a=primitive(slide,`${id}_core`,cx-13*k,cy-13*k,26*k,26*k,'roundRect','#E7F1FF',blue);
  primitive(slide,`${id}_inner`,cx-7*k,cy-7*k,14*k,14*k,'rect',blue,'none');
  for(let j=0;j<3;j++)for(const side of [0,1,2,3]){
    const q=(-8+j*8)*k;
    const x=side<2?cx+q:cx+(side===2?-19:13)*k;
    const y=side<2?cy+(side===0?-19:13)*k:cy+q;
    primitive(slide,`${id}_pin${side}_${j}`,x,y,side<2?2*k:6*k,side<2?6*k:2*k,'rect',blue,'none');
  }
  return a;
}
function queue(slide,id,x,y,k=1){
  const a=box(slide,`${id}_anchor`,x,y,82*k,27*k,'',12,'none','none');
  for(let j=0;j<3;j++)primitive(slide,`${id}_job${j}`,x+j*28*k,y,23*k,25*k,'roundRect',j===0?'#F6AB52':'#D8E8FF',j===0?'#F19637':light);
  return a;
}
function graph(slide,nodes){
  link(slide,nodes[2],nodes[3],'right','left',true,false,'#A9B6C8',false);
  link(slide,nodes[0],nodes[2],'left','left',true,false,'#A9B6C8',false);
  link(slide,nodes[1],nodes[3],'right','right',true,false,'#A9B6C8',false);
  link(slide,nodes[0],nodes[1],'right','left',false,false,'#EF942B',true);
}

const s4=deck.slides.add();
box(s4,'scene4_frame',600,190,635,455,'',20,'#FBFDFF',light);
label(s4,'scene4_arrival_label',800,204,240,32,'随机任务到达',21,blue,true);
const nodes4=[];
for(const [i,cx,cy] of [[1,745,322],[2,1100,322],[3,745,514],[4,1100,514]]){
  const d=drone(s4,`scene4_uav${i}`,cx,cy,1.05);nodes4.push(d);
  const job=task(s4,`scene4_task${i}`,cx-96,cy-77,.9);link(s4,job,d,'bottom','top');
  label(s4,`scene4_name${i}`,cx-65,cy+36,130,28,`UAV ${i}`,18,blue,true);
  chip(s4,`scene4_cpu${i}`,cx-35,cy+81,.65);queue(s4,`scene4_queue${i}`,cx-9,cy+71,.6);
}
graph(s4,nodes4);
label(s4,'scene4_link_label',812,397,220,48,'灰虚线：候选邻居\n橙箭头：单跳卸载',16,blue);

const s5=deck.slides.add();
const nodes5=[];
for(const [i,cx,cy] of [[1,100,483],[2,253,483],[3,100,590],[4,253,590]]){
  nodes5.push(drone(s5,`scene5_uav${i}`,cx,cy,.52));
  label(s5,`scene5_name${i}`,cx-40,cy+18,80,25,`UAV ${i}`,13,blue);
}
graph(s5,nodes5);
label(s5,'scene5_task_label',40,435,266,27,'随机任务到达',16,blue,true);
label(s5,'scene5_footer',35,633,285,26,'橙箭头：单跳 U2U 卸载',14,ink);

const s10=deck.slides.add();
box(s10,'scene10_frame',605,83,640,584,'',20,'#FAFCFF','#A7BCE3');
label(s10,'scene10_arrival_label',804,96,240,32,'随机任务到达',21,blue,true);
const nodes10=[];
for(const [i,cx,cy,profile] of [[1,748,211,'均衡型'],[2,1101,211,'资源较弱型'],[3,748,372,'计算较强型'],[4,1101,372,'能量受限型']]){
  const d=drone(s10,`scene10_uav${i}`,cx,cy,1.0);nodes10.push(d);
  const job=task(s10,`scene10_task${i}`,cx-95,cy-67,.8);link(s10,job,d,'bottom','top');
  label(s10,`scene10_name${i}`,cx-100,cy+33,200,28,`UAV ${i}  ${profile}`,17,blue,true);
  chip(s10,`scene10_cpu${i}`,cx-35,cy+76,.6);queue(s10,`scene10_queue${i}`,cx-10,cy+67,.58);
}
graph(s10,nodes10);
label(s10,'scene10_u2u_label',813,273,220,49,'灰虚线：候选邻居\n橙箭头：单跳卸载',15,blue);
// The lower pictorial branch shows processing alternatives, not extra UAVs.
const q=queue(s10,'scene10_route_queue',651,539,.95);
label(s10,'scene10_queue_label',623,575,136,27,'未绑定任务',17,blue);
const local=chip(s10,'scene10_local_cpu',936,512,1.2);
label(s10,'scene10_local_label',872,540,134,27,'本地 CPU',17,blue);
const remote=chip(s10,'scene10_remote_cpu',1137,603,1.2);
label(s10,'scene10_remote_label',1056,628,166,27,'邻居 UAV 的 CPU',16,blue);
label(s10,'scene10_tx_label',842,590,180,27,'单跳 U2U 传输',16,blue);
link(s10,q,local,'right','left');link(s10,q,remote,'right','left',false,false,'#EF942B');

// Reuse the original four-layer composition without the unmodeled tower.
const s8=deck.slides.add();
for(const [i,y,c] of [[0,85,'#C5DDFF'],[1,205,'#87B8FA'],[2,325,'#4B94E8'],[3,445,'#1558B5']]){
  primitive(s8,`scene8_layer${i}`,454,y,370,190,'diamond',c,'none');
}
drone(s8,'scene8_mobility_uav',640,161,.65);
label(s8,'scene8_mobility_label',540,190,200,31,'外生移动',18,blue,true);
const graph8=[];
for(const [i,x,y] of [[1,593,277],[2,687,277],[3,593,320],[4,687,320]]){
  graph8.push(primitive(s8,`scene8_neighbor${i}`,x-10,y-10,20,20,'ellipse','#FFFFFF',blue));
}
for(const [a,b] of [[0,1],[0,2],[1,3],[2,3]])link(s8,graph8[a],graph8[b],'right','left',false,false,'#FFFFFF',false);
label(s8,'scene8_graph_label',540,327,200,29,'局部邻居关系',17,blue,true);
chip(s8,'scene8_queue_cpu',605,414,1.05);queue(s8,'scene8_work_queue',640,402,.85);
label(s8,'scene8_queue_label',540,445,200,29,'任务队列与算力',17,'#FFFFFF',true);
primitive(s8,'scene8_feasibility_shield',613,513,54,61,'pentagon','none','#FFFFFF',180);
label(s8,'scene8_feasibility_check',615,518,50,44,'✓',29,'#FFFFFF',true);
label(s8,'scene8_feasibility_label',540,575,200,30,'联合约束',18,'#FFFFFF',true);

await fs.mkdir(out.slice(0,out.lastIndexOf('/')),{recursive:true});
await (await PresentationFile.exportPptx(deck)).save(out);
process.stdout.write(JSON.stringify({output:out,slides:deck.slides.items.length})+'\n');
