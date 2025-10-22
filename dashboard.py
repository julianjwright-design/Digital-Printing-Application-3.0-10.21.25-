
import streamlit as st, pandas as pd, numpy as np, datetime as dt
from pathlib import Path
from solver import build_and_solve, load_inputs
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

st.set_page_config(page_title='Digital Print Planner', layout='wide')
st.title('Digital Printing – ECP-Driven Planning Model')

DATA = Path('data'); OUT = Path('outputs')

def apply_sensitivity(products, labor, machines, materials, yield_pct=0.0, spoilage_pct=0.0, automation_pct=0.0, util_pct=0.0, price_pct=0.0):
    p = products.copy(); l = labor.copy(); k = machines.copy(); m = materials.copy()
    for col in ['Y_print','Y_finish']:
        p[col] = np.clip(p[col]*(1.0+yield_pct), 0.0001, 1.0)
    m['spoilage_rate'] = np.clip(m['spoilage_rate']*(1.0+spoilage_pct), 0.0, 0.95)
    l['automation_ratio'] = np.clip(l['automation_ratio']*(1.0+automation_pct), 0.0, 0.95)
    k['utilization_cap'] = np.clip(k['utilization_cap']*(1.0+util_pct), 0.1, 0.99)
    p['price'] = p['price']*(1.0+price_pct)
    return p,l,k,m

def generate_pdf(res, kpis):
    pdf_path = OUT/'summary_report.pdf'
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5,11)); plt.axis('off')
        title = 'Digital Printing – Planning Summary\n' + dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        plt.text(0.5,0.92,title,ha='center',fontsize=18)
        txt = (
            'Solver Status: ' + str(kpis.get('solver_status')) + '\n' +
            'Total Revenue: $' + f"{kpis.get('total_revenue',0):,.2f}" + '\n' +
            'Materials: $' + f"{kpis.get('total_material_cost',0):,.2f}" + '\n' +
            'Labor: $' + f"{kpis.get('total_labor_cost',0):,.2f}" + '\n' +
            'Machines: $' + f"{kpis.get('total_machine_cost',0):,.2f}" + '\n' +
            'Total Contribution: $' + f"{kpis.get('total_contribution',0):,.2f}" + '\n'
        )
        plt.text(0.1,0.75,'Key Performance Indicators',fontsize=14,weight='bold')
        plt.text(0.1,0.70,txt,fontsize=12,family='monospace')
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.5,11)); ax = plt.gca()
        rs = res.sort_values('contribution', ascending=False)
        ax.bar(rs['job_id'], rs['contribution'])
        ax.set_title('Contribution by Job'); ax.set_ylabel('USD'); ax.set_xlabel('Job')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.5,11)); ax = plt.gca()
        ax.scatter(res['unit_should_cost'], res['price'])
        for _, r in res.iterrows():
            ax.annotate(r['job_id'], (r['unit_should_cost'], r['price']), fontsize=9, xytext=(4,4), textcoords='offset points')
        ax.set_xlabel('Unit Should-Cost'); ax.set_ylabel('Price'); ax.set_title('Unit Should-Cost vs Price')
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return str(pdf_path)

with st.sidebar:
    st.header('Run Settings')
    objective = st.selectbox('Objective', ['max_margin','min_cost'], index=0)
    use_lots = st.checkbox('Enable lot-sizing (integer lots)', value=False)
    log_solver = st.checkbox('Show solver log', value=False)
    st.markdown('---')
    st.header('Sensitivity (±3%)')
    yield_delta = st.slider('Yield change', -0.03, 0.03, 0.00, 0.001)
    spoilage_delta = st.slider('Spoilage change', -0.03, 0.03, 0.00, 0.001)
    automation_delta = st.slider('Automation ratio change', -0.03, 0.03, 0.00, 0.001)
    util_delta = st.slider('Machine utilization cap change', -0.03, 0.03, 0.00, 0.001)
    price_delta = st.slider('Price change', -0.03, 0.03, 0.00, 0.001)
    st.caption('Sliders scale current inputs (e.g., +0.02 raises yields by 2%).')

col1, col2 = st.columns([1,1])

with col1:
    st.subheader('Inputs (current)')
    try:
        products, labor, machines, materials = load_inputs()
        st.expander('products.csv').dataframe(products, use_container_width=True, hide_index=True)
        st.expander('labor_standards.csv').dataframe(labor, use_container_width=True, hide_index=True)
        st.expander('machines.csv').dataframe(machines, use_container_width=True, hide_index=True)
        st.expander('materials.csv').dataframe(materials, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f'Error reading inputs: {e}')

with col2:
    st.subheader('Solve')
    if st.button('Run Planner with Sensitivity', type='primary'):
        try:
            p2,l2,k2,m2 = apply_sensitivity(products,labor,machines,materials,
                                            yield_pct=yield_delta, spoilage_pct=spoilage_delta,
                                            automation_pct=automation_delta, util_pct=util_delta, price_pct=price_delta)
            res,kpis,_ = build_and_solve(objective_mode=objective, integer_lots=use_lots, log=log_solver,
                                         df_products=p2, df_labor=l2, df_machines=k2, df_materials=m2)
            st.success('Solved. Status: ' + str(kpis['solver_status']))
            c1,c2,c3 = st.columns(3)
            c1.metric('Total Revenue', f"${kpis['total_revenue']:,.2f}")
            c2.metric('Total Contribution', f"${kpis['total_contribution']:,.2f}")
            c3.metric('Materials', f"${kpis['total_material_cost']:,.2f}")
            st.metric('Labor', f"${kpis['total_labor_cost']:,.2f}")
            st.metric('Machines', f"${kpis['total_machine_cost']:,.2f}")
            st.subheader('Results by Job')
            st.dataframe(res, use_container_width=True, hide_index=True)
            if st.button('Generate PDF Summary Report'):
                pdf_path = generate_pdf(res, kpis)
                st.success('PDF generated.')
                st.download_button('Download PDF summary', open(pdf_path,'rb').read(), file_name='summary_report.pdf')
        except Exception as e:
            st.error('Solver error: ' + str(e))

st.markdown('---')
st.caption('Tip: use sensitivity sliders to simulate CI gains (yield ↑, spoilage ↓, automation ↑).')
