import os

app_path = r"e:\PROJECTS\Advanced Derivatives Pricing\src\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i == 112: # Target: col1, col2, col3 = st.columns(3)
        new_lines.append('tab1, tab2 = st.tabs(["⚡ Live Execution Hub", "📉 Historical Black Swan Simulator"])\n\nwith tab1:\n')
    
    if i >= 112:
        if line.strip() == "":
            new_lines.append("\n")
        else:
            new_lines.append("    " + line)
    else:
        new_lines.append(line)

new_lines.append("\nwith tab2:\n")
new_lines.append("    st.header(\"📉 COVID-19 Historical Hedging Deviations\")\n")
new_lines.append("    st.markdown(\"Loading empirical arrays natively...\")\n")

with open(app_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Successfully injected Streamlit Tab Architecture.")
