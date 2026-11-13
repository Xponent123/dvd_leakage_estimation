#!/usr/bin/env python3
"""Create the typed report and figures from measured, frozen result artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from lusr.spice import ROOT,sha256


def load(path):return json.loads((ROOT/path).read_text())
def number(value,digits=3):
    if value is None:return 'n/a'
    value=float(value)
    if value==0:return '0'
    if abs(value)>=1e4 or abs(value)<.001:return f'{value:.2e}'
    return f'{value:.{digits}g}'


def figure_assets():
    out=ROOT/'reports/figures';out.mkdir(parents=True,exist_ok=True)
    for case in ('nmos_on','nmos_off','pmos_on','pmos_off'):
        frame=pd.read_csv(ROOT/'results/test_predictions'/f'{case}.csv')
        fig,axes=plt.subplots(2,2,figsize=(8.5,6.1),layout='constrained')
        for terminal,ax in zip(('id','ig','is','ib'),axes.flat):
            y=frame[terminal+'_spice'];p=frame[terminal+'_ml']
            threshold=max(float(np.median(np.abs(y)))*.01,1e-18)
            ax.scatter(y,p,s=3,alpha=.35,rasterized=True)
            lo=min(y.min(),p.min());hi=max(y.max(),p.max())
            ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
            ax.set(xscale='symlog',yscale='symlog',xlabel=f'SPICE {terminal} (A)',ylabel=f'ML {terminal} (A)')
            ax.set_xscale('symlog',linthresh=threshold);ax.set_yscale('symlog',linthresh=threshold)
            ax.tick_params(labelsize=7);ax.grid(alpha=.15)
        fig.suptitle(case.upper().replace('_',' ')+' — held-out test, signed currents')
        fig.savefig(out/f'{case}_parity.png',dpi=180);fig.savefig(out/f'{case}_parity.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.9),layout='constrained');ax.set_xlim(0,10);ax.set_ylim(-.3,8.3);ax.axis('off')
    for i in range(8):ax.text(.1,7.5-i,f'A{i}',ha='left',va='center',fontsize=11)
    nodes={}
    for i in range(4):nodes[f'p{i}']=(3,7-i*2)
    nodes.update({'p4':(5.8,6),'p5':(5.8,2),'out':(8.5,4)})
    for name,(x,y) in nodes.items():
        box=FancyBboxPatch((x-.55,y-.38),1.1,.76,boxstyle='round,pad=.03',fc='#e9f2fa',ec='#31688e')
        ax.add_patch(box);ax.text(x,y,'XOR',ha='center',va='center',fontsize=10)
    for i in range(4):
        for j in range(2):
            ax.annotate('',(2.42,7-2*i+.15*(1-2*j)),(.5,7.5-2*i-j),arrowprops={'arrowstyle':'->','lw':1})
    for a,b in [('p0','p4'),('p1','p4'),('p2','p5'),('p3','p5'),('p4','out'),('p5','out')]:
        x,y=nodes[a];xx,yy=nodes[b]
        ax.annotate('',(xx-.6,yy),(x+.6,y),arrowprops={'arrowstyle':'->','lw':1})
    ax.annotate('',(9.8,4),(9.1,4),arrowprops={'arrowstyle':'->'});ax.text(9.35,4.35,'Y',fontsize=11)
    ax.text(5,-.15,'Each XOR = 4 NAND2 cells. Total: 28 NAND2, 112 MOSFETs.',ha='center',fontsize=11)
    fig.savefig(out/'architecture.png',dpi=180);fig.savefig(out/'architecture.pdf');plt.close(fig)
    frame=pd.read_csv(ROOT/'results/circuit_predictions.csv')
    corners=pd.read_csv(ROOT/'data/corners.csv')
    errors=frame.groupby('corner_id').relative_error.apply(lambda x:100*np.mean(np.abs(x))).rename('mape').reset_index().merge(corners,on='corner_id')
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.6),layout='constrained')
    for ax,key,label in [(axes[0],'temp','Temperature (°C)'),(axes[1],'n_vth0','NMOS vth0 (V)')]:
        scatter=ax.scatter(errors[key],errors.mape,c=errors.vdd,cmap='viridis',s=18)
        ax.set(xlabel=label,ylabel='Circuit mean absolute relative error (%)');ax.grid(alpha=.2)
    fig.colorbar(scatter,ax=list(axes),label='VDD (V)')
    fig.savefig(out/'corner_error.png',dpi=180);fig.savefig(out/'corner_error.pdf');plt.close(fig)
    errors.to_csv(ROOT/'results/circuit_corner_errors.csv',index=False)


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT)
    args=parser.parse_args();ROOT=args.output_root.resolve()
    summary=load('results/hierarchy_summary.json');ml=load('results/ml_manifest.json')
    checks=load('reports/data_checks/data_checks.json');sim=load('reports/spice_both_manifest.json')
    audit=load('reports/circuit_reference_validation.json')
    decisions=load('docs/design_decisions.json');inputs=load('data/input_manifest.json')
    if ml['status']!='COMPLETED' or audit['status']!='PASSED':
        raise ValueError('Complete held-out evaluation and independent reference validation first')
    for case,metadata in ml['cases'].items():
        if sha256(ROOT/f'models/{case}.joblib')!=metadata['selected_model_sha256'] or summary['model_hashes'][case]!=metadata['selected_model_sha256']:
            raise ValueError(f'Stale selected model or hierarchy result: {case}')
    if 'ml_manifest_sha256' in summary and summary['ml_manifest_sha256']!=sha256(ROOT/'results/ml_manifest.json'):
        raise ValueError('Hierarchy evaluation used a different model manifest')
    figure_assets()
    revised='frozen_utc' in ml
    count=inputs['datasets']['nmos_on']['rows']
    partitions=ml['cases']['nmos_on']['provenance']['partition_counts']
    split_text=' / '.join(f'{n:,}' for n in (count,partitions['train'],partitions['validation'],partitions['test']))
    total=sum(d['rows'] for d in inputs['datasets'].values())
    val=pd.read_csv(ROOT/'results/transistor_validation_models.csv')
    test=pd.read_csv(ROOT/'results/transistor_test_metrics.csv')
    gate=pd.read_csv(ROOT/'results/gate_metrics.csv',dtype={'input_bits':str})
    source=ROOT/'reports/LUSR_8bit_Parity_Leakage_Report.md'
    pdf=source.with_suffix('.pdf')
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    if Path(font).exists():
        pdfmetrics.registerFont(TTFont('Body',font));pdfmetrics.registerFont(TTFont('BodyBold',bold))
        pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
        bodyfont='Body';boldfont='BodyBold'
    else:bodyfont='Helvetica';boldfont='Helvetica-Bold'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TextCustom',fontName=bodyfont,fontSize=9.2,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='HeadingCustom',fontName=boldfont,fontSize=15,leading=20,spaceBefore=8,spaceAfter=12,keepWithNext=True))
    styles.add(ParagraphStyle(name='TitleCustom',fontName=boldfont,fontSize=25,leading=32,spaceAfter=22))
    styles.add(ParagraphStyle(name='SmallCustom',fontName=bodyfont,fontSize=7,leading=10,spaceAfter=6))
    styles.add(ParagraphStyle(name='TableCustom',fontName=bodyfont,fontSize=7,leading=9))
    story=[];markdown=[]
    def text(s,style='TextCustom'):
        story.append(Paragraph(escape(s),styles[style]));markdown.append(s+'\n')
    def heading(s):
        story.append(Paragraph(escape(s),styles['HeadingCustom']));markdown.append('\n## '+s+'\n')
    def page():story.append(PageBreak())
    def table(headers,rows,widths=None):
        data=[[Paragraph(escape(str(x)),styles['TableCustom']) for x in row] for row in [headers,*rows]]
        t=Table(data,colWidths=widths or [505/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeaf2')),('VALIGN',(0,0),(-1,-1),'TOP'),
                             ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#62869e')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7fa')]),
                             ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,10)])
        mdcell=lambda value:str(value).replace('|',r'\|')
        markdown.append('| '+' | '.join(mdcell(x) for x in headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
        markdown.extend('| '+' | '.join(mdcell(x) for x in row)+' |' for row in rows);markdown.append('')
    def figure(relative,width=485,caption=None):
        from PIL import Image as PILImage
        path=ROOT/relative
        with PILImage.open(path) as im:w,h=im.size
        story.append(Image(str(path),width=width,height=width*h/w))
        if caption:text(caption,'SmallCustom')
        markdown.append(f'![{caption or path.stem}]({Path(relative).relative_to("reports")})\n')
    text('ML-Driven PVT Variation Aware\nLeakage Estimation'.replace('\n',' '),'TitleCustom')
    text('8-bit Parity Checker | 32nm_MGK | 0.9 V','HeadingCustom')
    text('Team 8 — LUSR')
    text('Dhimant Bhuva · Jal Parikh · Garima Mittal · Krishna Goel')
    text('Digital VLSI Design — Course Project. Results generated from the supplied model and the reproducible source package.')
    heading('Measured outcome')
    s=summary['circuit'];g=summary['gate']
    text(f'The project uses {total:,} single-transistor simulations to compare four regression families and selects one four-current model per transistor case. Gate validation covers all 18 input cases at 200 independent PVT corners; full parity validation covers all 256 vectors at those same corners (51,200 complete-netlist operating points).')
    table(['Evaluation','Median error (%)','95th-percentile (%)','Mean accuracy (%)'],[
        ['Five-gate aggregate',number(g['median_error_percent']),number(g['p95_error_percent']),f"{g['mean_accuracy_percent']:.4f}"],
        ['Complete parity checker',number(s['median_error_percent']),number(s['p95_error_percent']),f"{s['mean_accuracy_percent']:.4f}"]])
    attained=s['mean_accuracy_percent']>=99
    text(('The 99% mean-accuracy target was met for the complete circuit.' if attained else 'The 99% mean-accuracy target was not met for the complete circuit. The report retains the measured errors, including unfavorable results; no accuracy values are substituted or clipped.')+' Mean accuracy here is 100 × (1 − mean absolute relative error), not R².')
    text(f'All {total+54800:,} reference simulations completed and every gate/circuit output passed the digital logic-level checks. Additional fresh-netlist, current-balance and source-power checks are reported below.')
    if revised:
        text('The initial 10,000-row-per-case baseline reached 94.2965% circuit mean accuracy and is retained as a historical experiment. This revision follows the brief’s explicit permission to increase the sample count. Its 100,000-row-per-case datasets and 200 evaluation corners use a fresh seed, and its test partition was held back until the revised models were frozen. Earlier test results are not presented as independent verification of the revised models.')
    page();heading('1. Assignment, design and conventions')
    table(['Item','Implemented value'],[
        ['Assigned model','Original ../models/32nm_MGK.pm; SHA256 '+sim['model_sha256']],
        ['Supply and process','VDD 0.9 V nominal, uniform 0.81–0.99 V; five specified process parameters independently ±10%'],
        ['Geometry','Drawn L=32 nm; training W=32–256 nm; fixed gate NMOS W=64 nm, PMOS W=128 nm'],
        ['Temperature','Uniform −40 to 125 °C; model nominal temperature 27 °C'],
        ['Simulator','ngspice 42; model level=54, version=4.0, handled by ngspice BSIM4v5 implementation'],
        ['Solver options','gmin=1e-18; abstol=1e-18 A; reltol=1e-6; vntol=1e-9 V'],
        ['Parity convention','Eight total received bits. Y=A0 XOR … XOR A7; Y=1 means odd parity/error for an expected even-parity word.']],[110,395])
    text('The team authorized documented choices wherever the brief is silent. A balanced tree of seven XOR operations is implemented using four standard static-CMOS NAND2 cells per XOR: 28 NAND2 cells, 112 transistors. Widths are constant across gates; stacked transistors are not resized. This is a reproducible leakage-study design, with no claim of optimized delay. Diffusion areas/perimeters are left at model/simulator defaults because no layout was supplied.')
    figure('reports/figures/architecture.png',caption='A0 is the least-significant bit of vector integers; displayed words read A7…A0. Parity detects odd numbers of bit flips, not every multiple-bit error.')
    heading('2. Leakage observable and hierarchy')
    text('The primary observable is total static power supplied by every independent voltage source divided by VDD. With binary rail-valued inputs: I_leak = −I(VDD) − Σ I(Vin_high). This includes energy supplied through high input drivers when gate tunnelling is present. Supply-only −I(VDD) is also saved as a secondary diagnostic. Summing absolute transistor terminal currents would double-count paths and is not used.')
    text('Transistor currents are signed into their terminals and measured using four separate sources. For a source written Vd d 0 VD, Id=−I(Vd), with analogous definitions for gate, source and body. KCL requires Id+Ig+Is+Ib=0.')
    text('For ML energy aggregation, each transistor contributes P=(Vd−Vs)Id+(Vg−Vs)Ig+(Vb−Vs)Ib. This explicitly closes the source current as Is=−(Id+Ig+Ib). All four raw learned currents and their KCL residual are retained. Source closure avoids a common-mode power artifact from independently predicted currents; it does not enforce KCL at shared circuit nodes or guarantee positive predicted power.')
    text('Part B uses recorded SPICE internal/output voltages to form transistor features, exactly as directed by the brief. The voltage columns labelled for ML are copies with provenance, not independent ML voltage predictions. AND2 and OR2 references are complete connected NAND2+INV and NOR2+INV netlists. Part C propagates Boolean inputs through all 28 NAND2 cells, retrieves the matching gate prediction at the same corner, and sums those predictions; complete-circuit internal voltages and measured currents do not enter prediction.')
    heading('3. Data generation and checks')
    text(f'Sampling uses the fixed seed {inputs["random"]["master_seed"]} and independently generated streams. VDD is drawn before Vd and Vs, which are independent uniform draws conditional on VDD. W, temperature and all five process parameters are sampled independently. PMOS vth0 remains negative; its numeric interval endpoints are sorted. Every evaluation corner contains a shared NMOS process draw and a separate shared PMOS process draw.')
    table(['Case','Gate voltage','Body voltage','Rows / train / validation / test'],[
        ['NMOS ON','VDD','0',split_text],['NMOS OFF','0','0',split_text],
        ['PMOS ON','0','VDD',split_text],['PMOS OFF','VDD','VDD',split_text]],[95,65,65,280])
    if revised:
        text('A separate supplemental validation set contains 512 independently sampled single-FET PVT/width configurations per case, at exact and near supply rails. These are not added to training, not substituted for the uniform assigned datasets, and not drawn from gate/circuit corners. The seed is 2026102407. Its predefined selection score is MAPE of per-FET dissipated power/VDD on nonzero-bias configurations; the all-terminals-equal physical invariant is reported separately. All raw terminal-current errors remain available. This validation addresses the rail-boundary distribution shift without using any gate or circuit current for model selection.')
    text('ON/OFF are the brief’s gate-bias labels. Because source voltage varies independently, these labels do not guarantee strong inversion or cutoff at every sampled bias. Each CSV contains vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib, with currents in amperes and SI geometry.')
    table(['Case','Maximum |ΣI| (A)','Maximum |ΣI|/Σ|I|','Above mixed diagnostic threshold'],[
        [case,number(c['absolute_residual_A']['max']),number(c['normalized_residual']['max']),str(c['above_mixed_tolerance_count'])]
        for case,c in checks['transistor_datasets'].items()],[105,120,125,155])
    worst_kcl=max(c['normalized_residual']['max'] for c in checks['transistor_datasets'].values())
    text(f'The diagnostic threshold is 1e-18 A + 1e-6 × Σ|I|. Some finite-precision SPICE residuals exceed this diagnostic despite successful operating-point convergence. They are reported and retained, not projected out of the reference dataset. The worst normalized residual is {100*worst_kcl:.5g}%. Spread-out rows were additionally compared with independently instantiated fresh model cards to verify batched parameter updates.')
    text('Varying TOXE alone while retaining supplied TOXP and DTOX can trigger the BSIM warning that DTOX is ignored. TOXE remains active, as confirmed by a direct sensitivity experiment. The original model is unchanged; TOXP, TOXM, TOXREF and other unassigned parameters are never silently co-varied.')
    heading('Controlled physical trends')
    text('Each trend uses 41 points per polarity, with 246 extra fresh SPICE solves total. The illustrative baseline is L=W=32 nm, VDD=0.9 V and T=27 °C, with nominal process values. Only the horizontal-axis parameter changes. This illustrative W does not change the 64/128 nm gate sizing.')
    figure('reports/data_checks/plots/off_current_vs_temperature.png',width=425,caption='OFF drain current generally increases with temperature at fixed bias. Other parameters are held constant.')
    figure('reports/data_checks/plots/off_current_vs_vth0.png',width=425,caption='Increasing threshold magnitude reduces OFF current. PMOS vth0 is plotted with its physical negative sign.')
    figure('reports/data_checks/plots/gate_current_vs_toxe.png',width=425,caption='ON gate-current magnitude decreases as the electrical oxide thickness increases; supplied remaining oxide parameters are retained.')
    page();heading('4. Model comparison and held-out evaluation')
    families=ml.get('comparison_families',['random_forest','extra_trees','hist_gradient_boosting','mlp'])
    text('Four families are compared: '+', '.join(families)+'. Deterministic voltage, temperature and geometry transformations are computed from the 12 specified inputs. All fitted preprocessing uses training rows only. Validation controls model selection and documented tuning; the test partition is reserved for the selected models. No gate or circuit reference currents are used for fitting or selection.')
    text('Selection policy: '+ml['selection'])
    text('Preprocessing, hyperparameters and provenance are recorded in results/ml_manifest.json; revised training-run details are retained under physics_selected/. The saved model API accepts the original 12 inputs and returns four raw signed terminal currents. Reloaded predictions are checked against predictions before serialization.')
    if 'preprocessing' in ml:
        for key,value in ml['preprocessing'].items():text(key+': '+str(value),'SmallCustom')
    table(['Case / family / variant','Raw R²','Median error (%)','p95 error (%)','Boundary power MAPE (%)','Selected'],[
        [row.case+' / '+row.model+' / '+getattr(row,'variant','baseline'),number(row.raw_r2,9),number(row.median_relative_error_pct),number(row.p95_relative_error_pct),number(getattr(row,'boundary_power_mape_pct',None)),'yes' if row.selected else '']
        for row in val.itertuples()],[180,55,65,65,95,45])
    text('The 16-row table compares four families for each case. All rows use the same training and uniform-validation partitions. Raw-current statistics are reported in common units even when the training target representation differs. The selection score, which addresses the project’s power/leakage objective, is stated above; a family need not minimize every reported terminal-current statistic.')
    if revised:
        table(['Family','Main hyperparameters'],ml['hyperparameter_table'],[140,365])
        text(ml['representation_description'])
        text('In ordered, NMOS-polarity coordinates let ΔV=Vhigh−Vlow and H=(Ig+Ib)/2. The learned positive channel quantities are Gd=(Id+H)/ΔV and Gs=−(Is+H)/ΔV. Training divides these conductances and |Ig| by (W−10 nm)/100 nm; body current is not width normalized. Natural logarithms handle the dynamic range. The inverse transformation restores signs and the original drain/source ordering. Gd and Gs are separate learned outputs, so raw KCL remains a reported diagnostic.')
        text('For OFF cases, the gate-current baseline is the fixed BSIM gate-to-diffusion edge expression using the supplied card parameters; the model learns its logarithmic residual. The body baseline is 5e−15 × [2−exp(−Vlow/VT)−exp(−Vhigh/VT)] A. This body scale and the junction-coordinate warp are numerical conditioning choices, not exact BSIM body equations. For ON cases, an additional local RBF learns log(|Ig|/|Iedge_source+Iedge_drain|); half of its gate-current change is compensated in each drain/source prediction, preserving the learned channel difference and raw KCL residual. At zero edge voltage, tiny gate-body tunnelling is omitted by this ON gate baseline; its raw-current error remains in boundary diagnostics. The derivation, fixed coefficients and official references are in experiments/physics/RESEARCH.md.')
        if (ROOT/'results/selected_boundary_validation.csv').exists():
            boundary=pd.read_csv(ROOT/'results/selected_boundary_validation.csv')
            table(['Selected case','Boundary power MAPE (%)','Boundary median (%)','Boundary p95 (%)'],[
                [r.case,number(r.boundary_power_mape_pct),number(r.boundary_power_median_pct),number(r.boundary_power_p95_pct)] for r in boundary.itertuples()])
    else:
        table(['Family','Main hyperparameters'],[
        ['Random forest','320 trees, all features, unrestricted depth, minimum leaf size 1.'],
        ['Extra trees','400 trees, all features, unrestricted depth, minimum leaf size 1.'],
        ['Histogram gradient boosting','Separate regressor per target; 500 iterations, learning rate 0.08, 63 leaves, minimum leaf size 15, L2=0.01.'],
        ['MLP','Hidden layers 160/160/80; Adam, batch size 256, initial learning rate 0.001. Baseline ReLU, L2=1e-4, max 1,200 epochs; tuned ReLU/tanh, L2=1e-6, max 1,800. Early stopping uses only an internal split of training rows.']],[140,365])
        text('Tuned conductance candidates use the invertible training transform Gd=[Id+(Ig+Ib)/2]/(Vd−Vs), Gs=−[Is+(Ig+Ib)/2]/(Vd−Vs), along with Ig and Ib. Predictions reconstruct Id=(Vd−Vs)Gd−(Ig+Ib)/2 and Is=−(Vd−Vs)Gs−(Ig+Ib)/2. Random training rows have nonzero Vd−Vs; at equal terminal voltages, the reconstruction is a documented extrapolation. Both conductance outputs are learned separately. A small asinh scale gives low-current targets more weight; transformed targets are standardized using training rows only.')
    selected_validation=val[val.selected.astype(str).str.lower()=='true']
    table(['Selected case / variant','Validation MAPE (%)','Validation mean accuracy (%)'],[
        [row.case+' / '+row.model+' / '+getattr(row,'variant','baseline'),number(row.mape_pct),number(row.mean_accuracy_pct)] for row in selected_validation.itertuples()],[285,110,110])
    text('R² is computed separately for each terminal in raw amperes and uniformly averaged in the summary. It is displayed to nine significant digits; values extremely close to one are rounded. Relative errors pool the four terminal arrays. Error=|ML−SPICE|/max(|SPICE|,1e-18 A); very small signed currents can therefore produce large percentage errors. Per-terminal tables, absolute MAE and RMSE, denominator-floor counts and raw KCL diagnostics are supplied in the CSV results.')
    table(['Held-out case / selected family','R²','Median error (%)','95th-percentile (%)'],[
        [row.case+' / '+row.model,number(row.raw_r2,9),number(row.median_relative_error_pct),number(row.p95_relative_error_pct)]
        for row in test.itertuples()],[230,65,100,110])
    if revised:
        table(['Held-out case','Pooled terminal-current MAPE (%)','Single-FET power/VDD MAPE (%)'],[
            [r.case,number(r.mape_pct),number(r.power_mape_pct)] for r in test.itertuples()],[155,175,175])
        text('The project’s 99% target is assessed against circuit leakage, with the same defined source-power observable used throughout. It is not a claim that every signed terminal current has less than 1% relative error. OFF-current cancellation and tiny body currents can yield large raw-current percentage errors even when their effect on dissipated power is small. Body currents have physical temperature/bias dependence as well as numerical sensitivity near cancellation. Both metrics are reported above; no terminal, corner or input vector is excluded to improve the circuit result.')
    for case in ('nmos_on','nmos_off','pmos_on','pmos_off'):
        page();heading(case.upper().replace('_',' ')+' — test predictions')
        figure(f'reports/figures/{case}_parity.png',caption=f'Each panel includes all {partitions["test"]:,} held-out rows. Symmetric-log axes retain negative current predictions and measured signs. Dashed line denotes equality. No point is removed for poor accuracy.')
        terminal=pd.read_csv(ROOT/'results/transistor_test_terminals.csv')
        terminal=terminal[terminal.case==case]
        table(['Terminal','R²','MAE (A)','Median error (%)','95th-percentile (%)'],[
            [r.terminal,number(r.raw_r2,9),number(r.mae_A),number(r.median_relative_error_pct),number(r.p95_relative_error_pct)] for r in terminal.itertuples()])
    page();heading('5. Gate-level validation')
    text('Each row below summarizes 200 independent corners. SPICE/ML medians are expressed in nA; bias is mean signed ML−SPICE in nA. The same source-power definition is used on both sides. These 3,600 cases are evaluation-only.')
    table(['Gate / input','SPICE median (nA)','ML median (nA)','Bias (nA)','Median error (%)','p95 error (%)','Mean accuracy (%)'],[
        [r.gate+' '+r.input_bits,number(r.spice_median_A*1e9),number(r.ml_median_A*1e9),number(r.bias_A*1e9),number(r.median_error_percent),number(r.p95_error_percent),f'{r.mean_accuracy_percent:.4f}']
        for r in gate.itertuples()],[75,75,75,65,65,70,80])
    best=gate.loc[gate.median_error_percent.idxmin()];worst=gate.loc[gate.median_error_percent.idxmax()]
    text(f'By median relative error, the most accurate case is {best.gate} {best.input_bits} ({number(best.median_error_percent)}%); the least accurate is {worst.gate} {worst.input_bits} ({number(worst.median_error_percent)}%). The aggregate mean accuracy is {number(g["mean_accuracy_percent"])}%.')
    if gate.mean_accuracy_percent.min()>=99:
        lowest=gate.loc[gate.mean_accuracy_percent.idxmin()]
        text(f'All 18 gate/input cases individually exceed 99% mean accuracy over their 200 corners. The lowest is {lowest.gate} {lowest.input_bits}, at {lowest.mean_accuracy_percent:.6f}%.')
    figure('reports/figures/gate_parity.png',width=450,caption='All positive gate predictions on logarithmic axes. Any nonpositive predictions are counted explicitly and remain in all numerical metrics.')
    page();heading('6. Complete parity checker')
    text('The SPICE reference is one connected 112-transistor netlist, with all 256 patterns at each of the same 200 corners. Every output was checked against the independently propagated Boolean parity and the 0.2VDD/0.8VDD low/high limits. The complete 256-row per-vector metric table is results/circuit_metrics_by_vector.csv.')
    table(['Comparison','Median error (%)','p95 error (%)','Mean accuracy (%)','R²'],[
        [name,number(summary[key]['median_error_percent']),number(summary[key]['p95_error_percent']),f"{summary[key]['mean_accuracy_percent']:.4f}",number(summary[key]['r2'],9)]
        for name,key in [('Hierarchical ML vs full SPICE','circuit'),('Standalone SPICE gate sum vs full SPICE','ideal_gate_composition_only'),('Hierarchical ML vs standalone SPICE sum','ml_vs_standalone_gate_sum')]], [190,80,80,90,65])
    text('The second row is an evaluation-only diagnostic: it sums true standalone gate leakages using the same Boolean states. It quantifies error introduced by ideal-rail inputs and ignored inter-gate loading even if the transistor surrogate were perfect. It is never substituted for an ML prediction. The third row isolates the surrogate’s discrepancy from the standalone-gate reference sum.')
    text(f'Complete-circuit SPICE median={number(s["spice_median_A"]*1e9)} nA; ML median={number(s["ml_median_A"]*1e9)} nA; signed mean bias={number(s["bias_A"]*1e9)} nA. Nonpositive circuit predictions: {s["nonpositive_predictions"]}.')
    figure('reports/figures/circuit_parity.png',width=450,caption='Complete-circuit comparison over 51,200 observations. Logarithmic binning shows sample density; all observations remain in the result CSV.')
    page();heading('7. Error analysis and numerical validation')
    worst_rows=pd.read_csv(ROOT/'results/worst_gate_cases.csv',dtype={'input_bits':str}).head(5)
    table(['Gate / input','Corner','Temperature (°C)','VDD (V)','Absolute relative error (%)'],[
        [r.circuit+' '+r.input_bits,str(r.corner_id),number(r.temp),number(r.vdd),number(100*abs(r.relative_error))]for r in worst_rows.itertuples()])
    text('The largest-error rows and their corner identifiers are retained in results/worst_gate_cases.csv and worst_circuit_cases.csv. Full corner records provide both polarities’ process values. The scatter below shows association with PVT parameters, not a controlled causal experiment: other sampled parameters vary simultaneously.')
    figure('reports/figures/corner_error.png',caption='Each point summarizes 256 complete-circuit patterns at one corner. Color encodes VDD.')
    closure=summary['source_closure']
    text(f'Raw predicted four-current KCL residual: median {number(closure["median_raw_fet_kcl_A"])} A, maximum {number(closure["max_raw_fet_kcl_A"])} A across gate-device queries. Before any clipping (none is applied), {closure["negative_fet_power_count"]} FET predictions and {closure["negative_gate_power_count"]} whole-gate predictions have negative calculated power. Source closure ensures per-device current conservation for energy calculation, but does not enforce passivity or inter-device node balance.')
    text('The principal distribution shift is between uniformly sampled individual terminal voltages and gates, whose conducting transistor terminals often lie extremely close together or to a supply rail. Small-current targets and the wide current dynamic range make raw-space R² insufficient to establish percent accuracy. Errors from individual terminals can be amplified by a low reference leakage; current signs and near-zero denominators are therefore retained rather than concealed.')
    text('Independent validation artifacts: reports/circuit_reference_validation.json and its script/logs. These checks use separate terminal ammeters and manually assembled gate references to compare transistor dissipated power with external-source power, plus reverse-order/fresh-input parity checks. Numerical verdict: '+str(audit.get('status',audit.get('passed','See detailed validation JSON')))+'.')
    heading('8. Limitations and improvements')
    text('This is a compact-model simulation study, not measured silicon. The assigned predictive model and its ngspice implementation limit physical fidelity. Default diffusion geometry and no extracted interconnect parasitics mean layout leakage and loading are not modeled. The independent ±10% process draws are the assignment’s statistical experiment, not a calibrated foundry process distribution.')
    text('The prescribed Part B method already needs SPICE internal/output voltages at each evaluated gate corner; it is not an entirely SPICE-free node solver. Part C then replaces real inter-stage analog behavior with Boolean states. Consequently a high transistor R² does not imply 99% gate or circuit accuracy.')
    if revised:
        text('Remaining improvements include better treatment of tiny body currents and signed cancellations, a gate-body residual that remains representable at zero edge voltage, and measured uncertainty on new PVT domains. The physics normalization is specific to the assigned model and is not claimed to transfer unchanged to another technology. Any targeted rail-boundary training would need to remain distinct from the brief’s uniform sampling requirement. An independently validated internal-node solver would be a separate extension. Gate/circuit evaluation currents must remain excluded from fitting and model selection.')
    else:
        text('Further work should allocate additional single-device training samples using validation-driven error analysis, improve signed-current representations, and compare smooth kernel regressors under the same held-out protocol. Evaluation gate/circuit currents must remain excluded from training and model selection.')
    heading('9. Reproduction and deliverable map')
    pipeline='scripts/run_revision.sh' if revised else 'scripts/run_all.sh'
    text(f'From the project directory, run bash scripts/00_setup.sh, then bash {pipeline}. The setup is local to the project; no system package installation is required. The ordered pipeline samples, simulates devices, trains models, checks trends, simulates gates/circuit, validates references, evaluates hierarchy, builds this report, and checks the final deliverables. Fixed seeds, CSV splits, exact dependency versions and file checksums are included. Cached successful reference simulations are keyed to their inputs and model/executable hashes.')
    table(['Deliverable','Location'],[
        [f'Four {count:,}-row datasets','data/transistors/{case}.csv'],['Fixed partitions and 200 corners','data/splits/ and data/corners.csv'],
        ['Example decks','netlists/transistors/, netlists/gates/, netlists/circuit/parity8.cir'],
        ['KCL / three controlled trends','reports/data_checks/'],['16-row model comparison','results/transistor_validation_models.csv'],
        ['Four saved models','models/{case}.joblib'],['Held-out currents and metrics','results/test_predictions/ and transistor_test*.csv'],
        ['18-case gate table and predictions','results/gate_metrics.csv and gate_predictions.csv'],
        ['51,200 circuit predictions / 256-vector table','results/circuit_predictions.csv and circuit_metrics_by_vector.csv'],
        ['Source code / environment','lusr/, scripts/, requirements.lock.txt, docs/setup.md'],
        ['Typed report / figures','reports/LUSR_8bit_Parity_Leakage_Report.pdf and .md; reports/figures/']],[185,320])
    heading('References')
    for ref in [
        'Supplied Digital VLSI Design course-project brief: DVD_ML_PROJECT.pdf, pp. 1–4.',
        'Supplied Project-1_leakage_estimation_annotated.pdf. Slide 15 explicitly labels an illustrative equation incomplete; it is not used as a complete leakage formula.',
        'ngspice 42 User Manual: https://ngspice.sourceforge.io/docs/ngspice-42-manual.pdf',
        'Official ngspice control-language tutorial: https://ngspice.sourceforge.io/ngspice-control-language-tutorial.html',
        'BSIM4 User Manual, oxide definitions: https://ngspice.sourceforge.io/external-documents/models/BSIM480_Manual.pdf',
        'scikit-learn, common pitfalls and data leakage: https://scikit-learn.org/stable/common_pitfalls.html',
        'scikit-learn, mean absolute percentage error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html',
        'SciPy, RBFInterpolator (local neighbors and polynomial degree requirements): https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html',
    ]:text(ref,'SmallCustom')
    source.write_text('\n'.join(markdown)+'\n')
    def footer(canvas,doc):
        canvas.setFont(bodyfont,7);canvas.setFillColor(colors.HexColor('#62707c'))
        canvas.drawString(45,24,'LUSR | 32nm_MGK | 0.9 V | 8-bit parity checker')
        canvas.drawRightString(A4[0]-45,24,str(doc.page))
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=42,
                      title='LUSR — ML-driven PVT-aware leakage estimation',author=', '.join(decisions['members'])).build(story,onFirstPage=footer,onLaterPages=footer)
    print(pdf)


if __name__=='__main__':main()
