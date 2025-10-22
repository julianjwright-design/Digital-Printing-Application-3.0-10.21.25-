
import pandas as pd, numpy as np
import pulp as pl
from pathlib import Path
import json

DATA = Path('data')
OUT = Path('outputs')
OUT.mkdir(parents=True, exist_ok=True)

def _load_csvs():
    products = pd.read_csv(DATA/'products.csv')
    labor = pd.read_csv(DATA/'labor_standards.csv')
    machines = pd.read_csv(DATA/'machines.csv')
    materials = pd.read_csv(DATA/'materials.csv')
    return products, labor, machines, materials

def load_inputs(products=None, labor=None, machines=None, materials=None):
    if products is None or labor is None or machines is None or materials is None:
        p0, l0, k0, m0 = _load_csvs()
        products = p0 if products is None else products
        labor = l0 if labor is None else labor
        machines = k0 if machines is None else machines
        materials = m0 if materials is None else materials
    products = products.copy()
    products['Y'] = products['Y_print'] * products['Y_finish']
    if (products['Y'] <= 0).any() or (products['Y'] > 1).any():
        raise ValueError('Composite yield Y must be in (0,1].')
    return products, labor.copy(), machines.copy(), materials.copy()

def build_and_solve(objective_mode='max_margin', integer_lots=False, lot_size_cap=None, log=False,
                    df_products=None, df_labor=None, df_machines=None, df_materials=None):
    products, labor, machines, materials = load_inputs(df_products, df_labor, df_machines, df_materials)
    jobs = products['job_id'].tolist()
    prob = pl.LpProblem('DigitalPrintPlan', pl.LpMaximize if objective_mode=='max_margin' else pl.LpMinimize)
    x = pl.LpVariable.dicts('x', jobs, lowBound=0)
    g = pl.LpVariable.dicts('g', jobs, lowBound=0)

    b = {}
    if integer_lots:
        for _, mk in machines.iterrows():
            j = mk['job_id']; k = mk['machine']
            b[(k,j)] = pl.LpVariable(f'b_{k}_{j}', lowBound=0, cat='Integer')

    price = dict(zip(products['job_id'], products['price']))
    demand = dict(zip(products['job_id'], products['demand']))
    Y = dict(zip(products['job_id'], products['Y']))

    revenue = pl.lpSum(price[j]*x[j] for j in jobs)
    for j in jobs:
        prob += x[j] <= demand[j]
        prob += g[j] == x[j] / Y[j]

    mat_cost_terms = []
    for _, row in materials.iterrows():
        j = row['job_id']
        mat_cost_terms.append(float(row['qty_per_unit'])*(1.0+float(row['spoilage_rate']))*float(row['cost_per_unit'])*g[j])
    mat_cost = pl.lpSum(mat_cost_terms)

    roles = labor['role'].unique().tolist()
    labor_cost_terms = []
    for role in roles:
        sub = labor[labor['role']==role]
        role_rate = float(sub['std_rate'].iloc[0])
        avail_hr = float(sub['avail_hr'].iloc[0])
        util_tgt = float(sub['utilization_target'].iloc[0])
        role_time_terms = []
        for _, r in sub.iterrows():
            j = r['job_id']
            role_time_terms.append(float(r['base_hr_per_unit'])*(1.0-float(r['automation_ratio']))*g[j])
        total_role_hr = pl.lpSum(role_time_terms)
        prob += total_role_hr <= avail_hr*util_tgt
        labor_cost_terms.append(role_rate*total_role_hr)
    labor_cost = pl.lpSum(labor_cost_terms)

    mach_cost_terms = []
    for machine in machines['machine'].unique().tolist():
        sub = machines[machines['machine']==machine]
        avail = float(sub['avail_hr'].iloc[0])
        util_cap = float(sub['utilization_cap'].iloc[0])
        cost_rate = float(sub['cost_rate_hr'].iloc[0])
        u_terms = []
        for _, r in sub.iterrows():
            j = r['job_id']
            u_terms.append(float(r['hr_per_unit'])*g[j])
            if integer_lots:
                u_terms.append(float(r['setup_hr'])*b[(machine,j)])
        u_k = pl.lpSum(u_terms)
        prob += u_k <= avail*util_cap
        mach_cost_terms.append(cost_rate*u_k)
    mach_cost = pl.lpSum(mach_cost_terms)

    if objective_mode=='max_margin':
        prob += revenue - mat_cost - labor_cost - mach_cost
    else:
        for j in jobs: prob += x[j] == demand[j]
        prob += mat_cost + labor_cost + mach_cost

    prob.solve(pl.PULP_CBC_CMD(msg=1 if log else 0))

    status = pl.LpStatus[prob.status]
    res = pd.DataFrame({
        'job_id': jobs,
        'qty_planned': [pl.value(x[j]) for j in jobs],
        'gross_qty': [pl.value(g[j]) for j in jobs],
        'demand': [demand[j] for j in jobs],
        'price': [price[j] for j in jobs],
        'yield_composite': [Y[j] for j in jobs]
    })
    # Costs by job (approx allocations)
    mat_by_job = {j:0.0 for j in jobs}
    for _, row in materials.iterrows():
        j = row['job_id']
        mat_by_job[j] += float(row['qty_per_unit'])*(1.0+float(row['spoilage_rate']))*float(row['cost_per_unit'])*float(res.loc[res.job_id==j,'gross_qty'])
    res['mat_cost'] = res['job_id'].map(mat_by_job)

    lab_by_job = {j:0.0 for j in jobs}
    for role in roles:
        sub = labor[labor['role']==role]
        role_rate = float(sub['std_rate'].iloc[0])
        for _, r in sub.iterrows():
            j = r['job_id']
            lab_by_job[j] += role_rate*float(r['base_hr_per_unit'])*(1.0-float(r['automation_ratio']))*float(res.loc[res.job_id==j,'gross_qty'])
    res['labor_cost'] = res['job_id'].map(lab_by_job)

    mach_by_job = {j:0.0 for j in jobs}
    for machine in machines['machine'].unique().tolist():
        sub = machines[machines['machine']==machine]
        cost_rate = float(sub['cost_rate_hr'].iloc[0])
        for _, r in sub.iterrows():
            j = r['job_id']
            mach_by_job[j] += cost_rate*float(r['hr_per_unit'])*float(res.loc[res.job_id==j,'gross_qty'])
    res['machine_cost'] = res['job_id'].map(mach_by_job)

    res['total_cost'] = res['mat_cost'] + res['labor_cost'] + res['machine_cost']
    res['revenue'] = res['qty_planned']*res['price']
    res['contribution'] = res['revenue'] - res['total_cost']
    res['unit_should_cost'] = res['total_cost']/res['qty_planned'].replace(0, np.nan)
    res['unit_contribution'] = res['contribution']/res['qty_planned'].replace(0, np.nan)

    res.to_csv(OUT/'plan_results.csv', index=False)
    kpis = {
        'solver_status': status,
        'total_revenue': float(res['revenue'].sum()),
        'total_material_cost': float(res['mat_cost'].sum()),
        'total_labor_cost': float(res['labor_cost'].sum()),
        'total_machine_cost': float(res['machine_cost'].sum()),
        'total_contribution': float(res['contribution'].sum())
    }
    (OUT/'kpis.json').write_text(json.dumps(kpis, indent=2))
    return res, kpis, {}
