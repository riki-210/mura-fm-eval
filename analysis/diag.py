import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
B='#e7f1fb';Y='#f9eeda';G='#e5f4e4';P='#efedff';W='#ffffff'
def mk(w,h,xl,yl):
    fig,ax=plt.subplots(figsize=(w,h));ax.axis('off');ax.set_xlim(0,xl);ax.set_ylim(0,yl);return fig,ax
def box(ax,x,y,w,h,t,fc=B,fs=7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1',fc=fc,ec='#555555',lw=0.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(ax,x1,y1,x2,y2,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,color='#555555',lw=0.8,linestyle=ls))
# ---------- model figure
fig,ax=mk(9.4,5.2,100,56)
ax.text(1,54,'(a) Image-level DINOv2 model',fontsize=8.5,fontweight='bold');ax.text(52,54,'(b) Study-level multiple-instance model',fontsize=8.5,fontweight='bold')
# (a)
box(ax,1,38,13,9,'Radiograph\n(one view)');box(ax,17,38,15,9,'Border crop,\nCLAHE,\n224 × 224',G)
box(ax,35,38,13,9,'DINOv2-base\nencoder\n(ViT-B/14)')
arr(ax,14.3,42.5,16.7,42.5);arr(ax,32.3,42.5,34.7,42.5)
box(ax,1,24,22,9,'[class token ; mean\npatch token]',B);box(ax,26,24,22,9,'LayerNorm, dropout 0.3,\nlinear → p(abnormal)',B)
arr(ax,41.5,38,41.5,35);ax.plot([41.5,12],[35,35],color='#555555',lw=0.8);arr(ax,12,35,12,33.3);arr(ax,23.3,28.5,25.7,28.5)
box(ax,1,8,47,13,'Phase 1 (5 epochs): encoder frozen, head lr 1e-3\nPhase 2: all layers, AdamW (encoder 2e-5, head 2e-4,\nweight decay 0.05), warm-up + cosine, patience 8;\nclass-weighted cross-entropy, bfloat16',Y,fs=6.6)
arr(ax,37,24,37,21.3,ls='--')
# (b)
for i,x in enumerate([52,60.5,69]):box(ax,x,43,7,5,f'View {i+1}' if i<2 else 'View K ≤ 5',fs=6.5)
ax.text(66.3,45.5,'…',fontsize=9)
box(ax,52,34,24,6.5,'Shared DINOv2 encoder\n+ LayerNorm per view',B,fs=6.6)
for x in (55.5,64,72.5):arr(ax,x,43,x,40.3)
box(ax,80,43,19,5,'View dropout 0.15\n(training only)',Y,fs=6.6)
box(ax,52,25,24,7,'Pooling over views: gated attention\n(256 units) Σ a_k h_k, and max pooling\nmax_k h_k',B,fs=6.4)
arr(ax,64,34,64,32.4)
box(ax,80,25,19,7,'concat → LayerNorm,\ndropout 0.3, linear\n→ p(study abnormal)',B,fs=6.4)
arr(ax,76.3,28.5,79.7,28.5)
box(ax,52,8,47,13,'Initialised from the image-level checkpoint of the same seed;\nonly loss: class-weighted cross-entropy on the study label\n(no per-view loss); fine-tuning lr encoder 1e-5,\nother layers 1e-4; gradients accumulated over 4 studies',Y,fs=6.6)
arr(ax,89.5,25,89.5,21.3,ls='--')
fig.savefig('fig_models.pdf',bbox_inches='tight');fig.savefig('fig_models.png',dpi=200,bbox_inches='tight')
# ---------- evaluation figure
fig,ax=mk(9.4,4.6,100,50)
box(ax,1,32,17,12,'Test predictions\nper run (stored\nonce): 3 seeds ×\nconfigurations');box(ax,22,32,17,12,'Average the views\nof each study\n(study-level\nprobability)')
box(ax,43,32,17,12,'Threshold: 0.5 or\nYouden index on\nvalidation studies\n(never on test)',Y);box(ax,64,32,15,12,'Accuracy, sens.,\nspec., PPV, NPV,\nF1, kappa, AUC',G)
box(ax,83,32,16,12,'Calibration:\nECE, Brier score,\nreliability diagram',G)
for a in [(18.3,38,21.7,38),(39.3,38,42.7,38),(60.3,38,63.7,38),(79.3,38,82.7,38)]:arr(ax,*a)
box(ax,22,12,24,11,'Bootstrap over test\nstudies (2,000):\n95% intervals',P);box(ax,50,12,24,11,'Paired bootstrap\n(10,000): matched seeds,\nsame studies',P);box(ax,78,12,21,11,'Holm and BH\ncorrection:\nprimary / exploratory',P)
arr(ax,71.5,32,62,23.3);arr(ax,71.5,32,35,23.3);arr(ax,74.3,17.5,77.7,17.5)
box(ax,1,1,17,8,'Label audit:\n5-fold patient-\ngrouped CV',Y,fs=6.6);box(ax,22,1,24,8,'Out-of-fold study\nprobabilities → confident\nlearning flags (≤ 10%)',Y,fs=6.6)
arr(ax,18.3,5,21.7,5)
box(ax,50,1,24,8,'Same rule on test\nstudies (ensemble\ntrained without test)',Y,fs=6.6);arr(ax,46.3,5,49.7,5)
fig.savefig('fig_eval.pdf',bbox_inches='tight');fig.savefig('fig_eval.png',dpi=200,bbox_inches='tight')
