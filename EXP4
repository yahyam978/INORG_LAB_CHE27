import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import zscore
from io import BytesIO

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Pfefferkorn Plasticity Number Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

def create_plot(df_clean, plasticity_number, title):
    """Generates the plot of Ho/H vs. % Water and saves it to a BytesIO object."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the clean data points
    ax.scatter(df_clean['% Water'], df_clean['Ho/H'], color='blue', label='Data Points', zorder=5)
    
    # Sort data for smooth curve fitting/plotting
    df_sorted = df_clean.sort_values(by='% Water')
    x = df_sorted['% Water']
    y = df_sorted['Ho/H']
    
    # Use linear interpolation to draw a smooth curve
    # This is a common practice for connecting points in this type of experimental plot
    interp_func = interp1d(x, y, kind='linear')
    x_new = np.linspace(x.min(), x.max(), 500)
    y_new = interp_func(x_new)
    ax.plot(x_new, y_new, color='gray', linestyle='--')

    # Draw the line for Ho/H = 3.3 and find the intersection (Plasticity Number)
    # 1. Horizontal line at Ho/H = 3.3
    ax.axhline(y=3.3, color='red', linestyle='-', label='$H_0/H = 3.3$')
    
    # 2. Vertical line at the Plasticity Number
    ax.axvline(x=plasticity_number, color='green', linestyle=':', label='Plasticity Number')

    # Highlight the intersection point
    ax.scatter(plasticity_number, 3.3, color='red', marker='o', s=100, zorder=10)

    # Annotate the Plasticity Number on the graph
    ax.text(plasticity_number, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
            f'PN: {plasticity_number:.2f} %', color='green', fontsize=12, 
            ha='left', bbox=dict(facecolor='white', alpha=0.7))
    
    # Labels and Title
    ax.set_xlabel('Water Content (%)')
    ax.set_ylabel('$H_0/H$')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--')
    
    # Save plot to a PDF in memory
    buf = BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    buf.seek(0)
    return buf

def calculate_plasticity_number(df):
    """Calculates the Plasticity Number using linear interpolation."""
    df_sorted = df.sort_values(by='% Water')
    x = df_sorted['% Water'].values
    y = df_sorted['Ho/H'].values
    
    # Create an interpolation function: % Water = f(Ho/H)
    # We invert the axes because we are finding x (water content) for a given y (Ho/H)
    try:
        f_interp = interp1d(y, x, kind='linear', fill_value="extrapolate")
        plasticity_number = f_interp(3.3)
        return plasticity_number
    except ValueError:
        st.error("Cannot calculate Plasticity Number: The required value of Ho/H=3.3 is outside the range of your data for linear interpolation.")
        return np.nan

def outlier_detection(df, column, threshold=2.0):
    """Detects outliers in a column using Z-score."""
    # Ensure numerical data
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=[column], inplace=True)
    
    # Calculate Z-score
    df['Z_Score'] = zscore(df[column])
    
    # Identify outliers
    outliers = df[np.abs(df['Z_Score']) > threshold]
    
    # Clean dataframe
    df_clean = df[np.abs(df['Z_Score']) <= threshold].drop(columns=['Z_Score'])
    
    return df_clean, outliers.drop(columns=['Z_Score'])

# --- Streamlit App Interface ---
st.title("🧱 Experiment 4: Pfefferkorn Plasticity Number")
st.markdown("""
This application calculates the $\\frac{H_0}{H}$ ratio, detects outliers in the readings, 
and determines the **Plasticity Number (PN)** of the clay-water mixture.
The PN is the **Water Content (%)** at which $\\frac{H_0}{H} = 3.3$.
""")

# --- Data Input Section ---
st.header("1. Enter Experimental Data")
st.info("Edit the table below with your measurements for Water, $H_0$, and $H$. The **% Water** is automatically calculated assuming 100g of dry clay.")

# Initial data from the provided image/PPT for demonstration
initial_data = {
    'Water (g)': [10.0, 15.0, 20.0, 25.0],
    '% Water': [10.0, 15.0, 20.0, 25.0],
    'H₀ (cm)': [3.6, 3.5, 3.4, 3.8],
    'H (cm)': [1.6, 1.0, 1.5, 2.7]
}
initial_df = pd.DataFrame(initial_data)

# Editable data table
edited_df = st.data_editor(
    initial_df,
    num_rows="dynamic",
    column_config={
        'Water (g)': st.column_config.NumberColumn("Water (g)", help="Mass of water added to 100g clay", format="%.1f"),
        '% Water': st.column_config.NumberColumn("% Water", help="Water % based on 100g dry clay (Same as Water (g))", format="%.1f", disabled=True),
        'H₀ (cm)': st.column_config.NumberColumn("H₀ (cm)", help="Initial height of the cylindrical clay mass", format="%.1f"),
        'H (cm)': st.column_config.NumberColumn("H (cm)", help="Final height of the clay mass after compression", format="%.1f"),
    },
    key="data_editor"
)

# Recalculate % Water based on the assumption of 100g dry clay
if not edited_df.empty and 'Water (g)' in edited_df.columns:
    edited_df['% Water'] = edited_df['Water (g)'].apply(lambda x: x / 100 * 100 if pd.notna(x) else np.nan)


# --- Calculations and Results ---
if not edited_df.empty and all(col in edited_df.columns for col in ['H₀ (cm)', 'H (cm)']):
    
    st.header("2. Calculations and Outlier Check")
    
    # Calculate Ho/H
    edited_df['Ho/H'] = edited_df['H₀ (cm)'] / edited_df['H (cm)']
    
    # Outlier Detection
    st.subheader("Outlier Detection on $H_0/H$ Values")
    # Using Z-score with a threshold of 2.0 (can be adjusted)
    df_clean, outliers = outlier_detection(edited_df.copy(), 'Ho/H', threshold=2.0)
    
    if not outliers.empty:
        st.warning("⚠️ **Outliers Detected and Removed (Z-Score > 2.0)**")
        st.dataframe(outliers[['% Water', 'H₀ (cm)', 'H (cm)', 'Ho/H']], hide_index=True)
        st.markdown(f"The following {len(outliers)} reading(s) were removed from the analysis.")
    else:
        st.success("✅ **No significant outliers detected** in $H_0/H$ values.")

    st.subheader("Cleaned and Calculated Data")
    st.dataframe(df_clean[['% Water', 'H₀ (cm)', 'H (cm)', 'Ho/H']], hide_index=True)
    
    # Calculate Plasticity Number
    if len(df_clean) >= 2:
        plasticity_number = calculate_plasticity_number(df_clean)
        
        if not np.isnan(plasticity_number):
            st.header("3. Results and Plot")
            st.success(f"**Calculated Plasticity Number (PN): {plasticity_number:.2f} % Water**")
            
            # Generate and display plot
            st.subheader("Plot of $H_0/H$ vs. Water Content (%)")
            plot_pdf_buffer = create_plot(
                df_clean, 
                plasticity_number, 
                'Pfefferkorn Plasticity Number Determination'
            )
            
            # Display plot (Streamlit automatically shows Matplotlib figure object)
            st.pyplot(plt.gcf()) 
            
            # Download button
            st.download_button(
                label="Download Plot as PDF",
                data=plot_pdf_buffer,
                file_name="plasticity_number_plot.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("The plasticity number could not be calculated. Ensure your $\\frac{H_0}{H}$ values span the target of 3.3 and you have enough data points (min 2).")

    else:
        st.warning("Please enter at least two valid data points to perform the calculation and plotting.")

else:
    st.warning("Please ensure your data table is not empty and contains valid numeric values for $H_0$ and $H$.")
