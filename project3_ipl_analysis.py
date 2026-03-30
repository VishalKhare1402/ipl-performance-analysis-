"""
=============================================================
PROJECT 3: IPL Team & Player Performance Analysis
Tools: Python — Pandas, NumPy, Matplotlib
Author: Vishal Khare
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────
# STEP 1: GENERATE REALISTIC IPL DATASET
# ─────────────────────────────────────────
print("=" * 55)
print("   IPL TEAM & PLAYER PERFORMANCE ANALYSIS")
print("=" * 55)

teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
         'Rajasthan Royals', 'Sunrisers Hyderabad']

batsmen = {
    'Mumbai Indians':              ['Rohit Sharma', 'Ishan Kishan', 'Suryakumar Yadav', 'Hardik Pandya'],
    'Chennai Super Kings':         ['MS Dhoni', 'Ruturaj Gaikwad', 'Devon Conway', 'Ambati Rayudu'],
    'Royal Challengers Bangalore': ['Virat Kohli', 'Faf du Plessis', 'Glenn Maxwell', 'Dinesh Karthik'],
    'Kolkata Knight Riders':       ['Shreyas Iyer', 'Venkatesh Iyer', 'Andre Russell', 'Nitish Rana'],
    'Delhi Capitals':              ['David Warner', 'Prithvi Shaw', 'Rishabh Pant', 'Axar Patel'],
    'Punjab Kings':                ['Shikhar Dhawan', 'Jonny Bairstow', 'Liam Livingstone', 'Sam Curran'],
    'Rajasthan Royals':            ['Jos Buttler', 'Sanju Samson', 'Yashasvi Jaiswal', 'Devdutt Padikkal'],
    'Sunrisers Hyderabad':         ['Kane Williamson', 'Abhishek Sharma', 'Rahul Tripathi', 'Nicholas Pooran'],
}

# Build match-level data (14 seasons x ~60 matches each)
matches_list = []
match_id = 1
seasons = list(range(2011, 2024))

for season in seasons:
    team_list = teams.copy()
    np.random.shuffle(team_list)
    match_pairs = [(team_list[i], team_list[i+1]) for i in range(0, len(team_list)-1, 2)]
    # Each team plays ~14 games in a season
    for _ in range(7):
        np.random.shuffle(team_list)
        for i in range(0, len(team_list)-1, 2):
            t1, t2 = team_list[i], team_list[i+1]
            winner = np.random.choice([t1, t2], p=[0.52, 0.48])
            toss_winner = np.random.choice([t1, t2])
            toss_decision = np.random.choice(['bat', 'field'], p=[0.35, 0.65])
            t1_score = np.random.randint(130, 220)
            t2_score = np.random.randint(120, 215)
            matches_list.append({
                'match_id': match_id, 'season': season,
                'team1': t1, 'team2': t2, 'winner': winner,
                'toss_winner': toss_winner, 'toss_decision': toss_decision,
                'team1_score': t1_score, 'team2_score': t2_score
            })
            match_id += 1

matches_df = pd.DataFrame(matches_list)

# Build delivery-level data (batting stats per player per match)
deliveries_list = []
for _, match in matches_df.iterrows():
    for team in [match['team1'], match['team2']]:
        players = batsmen[team]
        runs_budget = np.random.randint(130, 210)
        for player in players:
            runs = max(0, int(np.random.normal(runs_budget / 4, 18)))
            balls = max(runs, int(np.random.normal(runs * 1.2, 10)))
            fours = int(runs * np.random.uniform(0.08, 0.18))
            sixes = int(runs * np.random.uniform(0.03, 0.12))
            dismissed = np.random.choice([True, False], p=[0.75, 0.25])
            deliveries_list.append({
                'match_id': match['match_id'],
                'season': match['season'],
                'team': team,
                'batsman': player,
                'runs': runs,
                'balls_faced': balls,
                'fours': fours,
                'sixes': sixes,
                'dismissed': dismissed
            })

deliveries_df = pd.DataFrame(deliveries_list)

print(f"\n[1] Dataset created:")
print(f"    Matches     : {len(matches_df)}")
print(f"    Innings     : {len(deliveries_df)}")
print(f"    Seasons     : {matches_df['season'].min()} – {matches_df['season'].max()}")

# ─────────────────────────────────────────
# STEP 2: DATA CLEANING
# ─────────────────────────────────────────
print("\n[2] DATA CLEANING...")

# Check nulls
print(f"    Matches nulls   : {matches_df.isnull().sum().sum()}")
print(f"    Deliveries nulls: {deliveries_df.isnull().sum().sum()}")

# Add derived columns
deliveries_df['strike_rate'] = np.where(
    deliveries_df['balls_faced'] > 0,
    (deliveries_df['runs'] / deliveries_df['balls_faced'] * 100).round(2),
    0
)
matches_df['margin'] = abs(matches_df['team1_score'] - matches_df['team2_score'])
print("    ✔ Strike rate calculated")
print("    ✔ Win margin calculated")

# ─────────────────────────────────────────
# STEP 3: STATISTICAL ANALYSIS
# ─────────────────────────────────────────
print("\n[3] STATISTICAL ANALYSIS...")

# --- Batting stats per player ---
batting = deliveries_df.groupby('batsman').agg(
    Total_Runs=('runs', 'sum'),
    Innings=('match_id', 'count'),
    Balls_Faced=('balls_faced', 'sum'),
    Fours=('fours', 'sum'),
    Sixes=('sixes', 'sum'),
    Dismissals=('dismissed', 'sum'),
    Avg_Strike_Rate=('strike_rate', 'mean')
).reset_index()

batting['Batting_Average'] = np.where(
    batting['Dismissals'] > 0,
    (batting['Total_Runs'] / batting['Dismissals']).round(2),
    batting['Total_Runs']
)
batting['Strike_Rate'] = (batting['Total_Runs'] / batting['Balls_Faced'] * 100).round(2)

# Consistency metric: lower std deviation = more consistent
consistency = deliveries_df.groupby('batsman')['runs'].agg(['std', 'mean']).reset_index()
consistency.columns = ['batsman', 'Std_Dev', 'Mean_Runs']
batting = batting.merge(consistency, on='batsman')
batting['Consistency_Score'] = (batting['Mean_Runs'] / (batting['Std_Dev'] + 1)).round(2)

batting = batting.sort_values('Total_Runs', ascending=False)

# --- Team performance ---
total_played = pd.concat([matches_df['team1'], matches_df['team2']]).value_counts().reset_index()
total_played.columns = ['Team', 'Played']
wins = matches_df['winner'].value_counts().reset_index()
wins.columns = ['Team', 'Wins']
team_perf = total_played.merge(wins, on='Team')
team_perf['Win_Rate_%'] = (team_perf['Wins'] / team_perf['Played'] * 100).round(1)
team_perf['Losses'] = team_perf['Played'] - team_perf['Wins']
team_perf = team_perf.sort_values('Win_Rate_%', ascending=False)

# Avg team score
team_scores = pd.concat([
    matches_df[['team1', 'team1_score']].rename(columns={'team1': 'Team', 'team1_score': 'Score'}),
    matches_df[['team2', 'team2_score']].rename(columns={'team2': 'Team', 'team2_score': 'Score'})
])
avg_scores = team_scores.groupby('Team')['Score'].mean().round(1).reset_index()
avg_scores.columns = ['Team', 'Avg_Score']
team_perf = team_perf.merge(avg_scores, on='Team')

# Toss analysis
toss_wins = matches_df[matches_df['toss_winner'] == matches_df['winner']].shape[0]
toss_total = len(matches_df)
toss_win_rate = toss_wins / toss_total * 100

# Season trends
season_trend = matches_df.groupby('season').agg(
    Avg_Score=('team1_score', 'mean'),
    Total_Matches=('match_id', 'count')
).round(1).reset_index()

print(f"    ✔ Batting stats for {len(batting)} players computed")
print(f"    ✔ Team win rates calculated")
print(f"    ✔ Season trends (2011–2023) computed")

# ─────────────────────────────────────────
# STEP 4: KEY INSIGHTS
# ─────────────────────────────────────────
print("\n[4] KEY INSIGHTS")
print("-" * 42)
top_batsman = batting.iloc[0]
top_team = team_perf.iloc[0]
print(f"    Top Run Scorer    : {top_batsman['batsman']} ({int(top_batsman['Total_Runs'])} runs)")
print(f"    Best Strike Rate  : {batting.nlargest(1,'Strike_Rate').iloc[0]['batsman']}")
print(f"    Most Consistent   : {batting.nlargest(1,'Consistency_Score').iloc[0]['batsman']}")
print(f"    Best Win Rate     : {top_team['Team']} ({top_team['Win_Rate_%']}%)")
print(f"    Toss-Win Advantage: {toss_win_rate:.1f}% (toss winners go on to win)")
print(f"    Avg Match Score   : {matches_df['team1_score'].mean():.1f} runs")

# ─────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────
print("\n[5] GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0a0e1a')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

DARK_BG  = '#111827'
ACCENT1  = '#f97316'
ACCENT2  = '#3b82f6'
ACCENT3  = '#22c55e'
TEXT_CLR = '#e2e8f0'
MUTED    = '#6b7280'

ax_colors = [ACCENT1, ACCENT2, ACCENT3, '#a855f7', '#ec4899',
             '#14b8a6', '#eab308', '#f43f5e']

# Plot 1: Top 10 run scorers
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(DARK_BG)
top10 = batting.head(10).sort_values('Total_Runs')
colors_b = [ACCENT1 if i == len(top10)-1 else ACCENT2 for i in range(len(top10))]
bars = ax1.barh(top10['batsman'], top10['Total_Runs'], color=colors_b, height=0.6, edgecolor='none')
ax1.set_title('🏆 Top 10 Run Scorers (All Time)', color=TEXT_CLR, fontsize=13, fontweight='bold', pad=10)
ax1.tick_params(colors=TEXT_CLR, labelsize=9)
ax1.spines[:].set_visible(False)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar, val in zip(bars, top10['Total_Runs']):
    ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f'{int(val):,}', va='center', color=TEXT_CLR, fontsize=8)

# Plot 2: Team win rates
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(DARK_BG)
short_names = [t.split()[-1] if len(t.split()) > 1 else t for t in team_perf['Team']]
bars2 = ax2.bar(short_names, team_perf['Win_Rate_%'], color=ax_colors, edgecolor='none', width=0.6)
ax2.axhline(50, color=MUTED, linestyle='--', linewidth=1, alpha=0.6)
ax2.set_title('Team Win Rates (%)', color=TEXT_CLR, fontsize=12, fontweight='bold')
ax2.tick_params(colors=TEXT_CLR, labelsize=8)
ax2.spines[:].set_visible(False)
ax2.set_ylim(0, 70)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha='right')
for bar, val in zip(bars2, team_perf['Win_Rate_%']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', color=TEXT_CLR, fontsize=7.5, fontweight='bold')

# Plot 3: Strike Rate vs Average scatter
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(DARK_BG)
sc = ax3.scatter(batting['Batting_Average'], batting['Strike_Rate'],
                  c=batting['Total_Runs'], cmap='plasma', s=60, alpha=0.8, edgecolors='none')
ax3.set_xlabel('Batting Average', color=MUTED, fontsize=9)
ax3.set_ylabel('Strike Rate', color=MUTED, fontsize=9)
ax3.set_title('Strike Rate vs Batting Average', color=TEXT_CLR, fontsize=12, fontweight='bold')
ax3.tick_params(colors=TEXT_CLR, labelsize=8)
ax3.spines[:].set_visible(False)
cbar = plt.colorbar(sc, ax=ax3)
cbar.set_label('Total Runs', color=MUTED, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)

# Plot 4: Season-wise avg score trend
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(DARK_BG)
ax4.fill_between(season_trend['season'], season_trend['Avg_Score'],
                  alpha=0.25, color=ACCENT3)
ax4.plot(season_trend['season'], season_trend['Avg_Score'],
          color=ACCENT3, linewidth=2, marker='o', markersize=5)
ax4.set_title('Season-wise Avg Match Score', color=TEXT_CLR, fontsize=12, fontweight='bold')
ax4.tick_params(colors=TEXT_CLR, labelsize=8)
ax4.spines[:].set_visible(False)
ax4.set_xlabel('Season', color=MUTED, fontsize=9)
ax4.set_ylabel('Avg Score (Runs)', color=MUTED, fontsize=9)

# Plot 5: Top 8 most consistent players
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(DARK_BG)
top_consistent = batting.nlargest(8, 'Consistency_Score').sort_values('Consistency_Score')
ax5.barh(top_consistent['batsman'], top_consistent['Consistency_Score'],
          color=ACCENT1, height=0.6, edgecolor='none', alpha=0.9)
ax5.set_title('Most Consistent Batsmen', color=TEXT_CLR, fontsize=12, fontweight='bold')
ax5.tick_params(colors=TEXT_CLR, labelsize=8)
ax5.spines[:].set_visible(False)
ax5.set_xlabel('Consistency Score', color=MUTED, fontsize=9)

fig.suptitle('🏏  IPL Performance Analysis  |  2011–2023', color='white',
              fontsize=17, fontweight='bold', y=1.01)

plt.savefig('/home/claude/ipl_analysis.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0e1a')
plt.close()
print("    ✔ Charts saved: ipl_analysis.png")

# ─────────────────────────────────────────
# STEP 6: EXPORT RESULTS
# ─────────────────────────────────────────
print("\n[6] EXPORTING RESULTS...")

with pd.ExcelWriter('/home/claude/ipl_analysis_results.xlsx', engine='xlsxwriter') as writer:
    batting.round(2).to_excel(writer, sheet_name='Batting Stats', index=False)
    team_perf.to_excel(writer, sheet_name='Team Performance', index=False)
    season_trend.to_excel(writer, sheet_name='Season Trends', index=False)
    matches_df.head(100).to_excel(writer, sheet_name='Sample Matches', index=False)

# Save datasets
deliveries_df.to_csv('/home/claude/ipl_deliveries.csv', index=False)
matches_df.to_csv('/home/claude/ipl_matches.csv', index=False)
print("    ✔ Excel results saved: ipl_analysis_results.xlsx")
print("    ✔ CSVs saved: ipl_deliveries.csv, ipl_matches.csv")
print("\n✅ PROJECT 3 COMPLETE!\n")
