from mx_osint.config import Settings
settings = Settings()

from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

def print_business_metrics(df: pl.DataFrame, learned_weights: dict[str, dict[str, float]]) -> None:
    """Prints a single consolidated executive summary of geospatial, empirical OSINT, and B2B metrics."""
    total = len(df)
    if total == 0:
        print('⚠️ Empty DataFrame — check coordinates and radius.')
        return

    def _cov(sub_df: pl.DataFrame, col: str) -> tuple[int, float]:
        if len(sub_df) == 0 or col not in sub_df.columns:
            return (0, 0.0)
        cnt = sub_df.filter(pl.col(col).is_not_null() & (pl.col(col) != '')).height
        return (cnt, cnt / len(sub_df) * 100.0)
    targets = df.filter(pl.col('b2b_status') == 'target')
    rescued = df.filter(pl.col('is_rescued_partial'))
    partials = df.filter((pl.col('b2b_status') == 'partial') & ~pl.col('is_rescued_partial'))
    excluded = df.filter(pl.col('b2b_status') == 'no_target')
    actionable = df.filter((pl.col('b2b_status') == 'target') | pl.col('is_rescued_partial'))
    print('\n' + '=' * 78)
    print(f'📊 RESUMEN EJECUTIVO OSINT B2B — THE COFFEE BAJIO HUB (Radio: {settings.RADIUS_KM} km)')
    print('=' * 78)
    print(f"Total unidades económicas únicas (>= '{settings.MIN_EMPLOYEES_STRATA}'): {total:,}")
    avg_dist = df['distance_km'].mean()
    w1 = df.filter(pl.col('distance_km') <= 1.0).height
    w3 = df.filter(pl.col('distance_km') <= 3.0).height
    print(f'\n📍 Distribución Geográfica desde Anchor Hub ({settings.TARGET_LAT:.5f}, {settings.TARGET_LON:.5f}):')
    print(f'   • Distancia promedio: {avg_dist:.2f} km | Dentro de 1.0 km: {w1:,} ({w1 / total * 100:.1f}%) | Dentro de 3.0 km: {w3:,} ({w3 / total * 100:.1f}%)')
    print('\n🎯 Segmentación B2B (SCIAN Privado + Rescate Empírico INEGI/IMSS):')
    print(f'   • ✅ Targets Directos Privados:          {len(targets):>4,} ({len(targets) / total * 100:5.1f}%)')
    print(f'   • 🚀 Rescatados (Alto Salario/Ing.):     {len(rescued):>4,} ({len(rescued) / total * 100:5.1f}%)')
    print(f'   • ⚠️ Parciales Estándar:                 {len(partials):>4,} ({len(partials) / total * 100:5.1f}%)')
    print(f'   • ❌ No Targets (Comercio/Gob/Rest.):    {len(excluded):>4,} ({len(excluded) / total * 100:5.1f}%)')
    em_c, em_p = _cov(actionable, 'email')
    ph_c, ph_p = _cov(actionable, 'phone')
    wb_c, wb_p = _cov(actionable, 'website')
    total_payroll_m = actionable['est_monthly_payroll_mxn'].sum() / 1000000.0
    print('\n📧 Cobertura de Contacto & Masa Salarial ( Universo Accionable = Directos + Rescatados ):')
    print(f'   • Email: {em_c:>3,} ({em_p:5.1f}%) | Teléfono: {ph_c:>3,} ({ph_p:5.1f}%) | Website: {wb_c:>3,} ({wb_p:5.1f}%)')
    print(f'   • Masa Salarial Mensual Estimada (INEGI/IMSS): ${total_payroll_m:,.1f} M MXN/mes')
    print('\n⚖️ Pesos del Modelo 5D: CRITIC (Objetivo) vs. AdaBoost (Machine Learning):')
    for feat, w_dict in learned_weights.items():
        print(f"   • {feat:<22} | CRITIC: {w_dict['CRITIC_Objective'] * 100:5.1f}%  |  AdaBoost: {w_dict['AdaBoost_Learned'] * 100:5.1f}%")
    print('=' * 78)

def plot_summary_charts(geo_df: pl.DataFrame, scored_df: pl.DataFrame, output_dir: Path) -> None:
    """Renders 3 diagnostic subplots in a single figure (Distances, Strata, Top B2B Sectors)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    distances = geo_df['distance_km'].to_list()
    axes[0].hist(distances, bins=25, color='steelblue', edgecolor='black', alpha=0.75)
    for r, col in [(1.0, 'red'), (3.0, 'orange'), (5.0, 'green')]:
        axes[0].axvline(x=r, color=col, linestyle='--', label=f'{r:.0f} km')
    axes[0].set_title('Distribución de Distancias desde Anchor Hub', fontweight='bold')
    axes[0].set_xlabel('Distancia (km)')
    axes[0].set_ylabel('Negocios (Todos los estratos)')
    axes[0].legend()
    strata_counts = scored_df.group_by('employee_strata').agg(pl.len().alias('count')).to_pandas().set_index('employee_strata').reindex(STRATA_ORDER[2:], fill_value=0).reset_index()
    sns.barplot(data=strata_counts, x='employee_strata', y='count', hue='employee_strata', palette='YlOrRd', legend=False, ax=axes[1])
    axes[1].set_title('Negocios por Estrato de Personal', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Negocios')
    axes[1].tick_params(axis='x', rotation=30)
    sector_counts = scored_df.filter(pl.col('b2b_status').is_in(['target', 'partial'])).group_by(['b2b_sector', 'b2b_status']).agg(pl.len().alias('count')).sort('count', descending=True).head(10).to_pandas()
    sns.barplot(data=sector_counts, x='count', y='b2b_sector', hue='b2b_status', palette={'target': '#2ca02c', 'partial': '#ff7f0e'}, dodge=False, ax=axes[2])
    axes[2].set_title('Top Sectores B2B (Target vs Partial)', fontweight='bold')
    axes[2].set_xlabel('Negocios')
    axes[2].set_ylabel('')
    plt.tight_layout()
    chart_path = output_dir / 'denue_b2b_diagnostics.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
