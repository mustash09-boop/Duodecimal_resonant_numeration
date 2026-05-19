from music12.core.theoretical_rc_chains12 import (
    export_theoretical_chain_table_csv,
    export_theoretical_chain_table_txt,
)

def main():
    export_theoretical_chain_table_csv(
        r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration\Block004_data\RealPiano_1\00_sources\reports\theoretical_rc_chains_5A_to_11_1.csv"
    )

    export_theoretical_chain_table_txt(
        r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration\Block004_data\RealPiano_1\00_sources\reports\theoretical_rc_chains_5A_to_11_1.txt"
    )

    print("DONE: theoretical chains generated")

if __name__ == "__main__":
    main()