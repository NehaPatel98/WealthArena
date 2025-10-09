"""
ASX Symbols Database for WealthArena Trading System

This module contains all ASX-listed symbols organized by category for comprehensive
trading across the Australian Securities Exchange.
"""

# Complete ASX symbols list (300+ symbols)
ASX_SYMBOLS = [
    # A
    "14D", "1AD", "1AE", "1AI", "1CG", "1MC", "1TT", "29M", "360", "3DA", "3DP", "3PL", "4DS", "4DX", "4DXN", "5EA", "5GG", "5GN", "88E", "8CO", "8IH",
    "A11", "A1G", "A1M", "A1N", "A2M", "A3D", "A4N", "A8G", "AA2", "AAC", "AAI", "AAJ", "AAL", "AAM", "AAP", "AAR", "AAU", "ABB", "ABE", "ABG", "ABV", "ABX", "ABY",
    "ACE", "ACF", "ACL", "ACM", "ACP", "ACQ", "ACR", "ACS", "ACU", "ACW", "AD1", "AD8", "ADC", "ADD", "ADG", "ADH", "ADN", "ADO", "ADR", "ADS", "ADV", "ADX", "ADY",
    "AEE", "AEF", "AEI", "AEL", "AER", "AEV", "AFA", "AFG", "AFI", "AFL", "AFP", "AGC", "AGD", "AGE", "AGH", "AGI", "AGL", "AGN", "AGR", "AGY", "AHC", "AHF", "AHI", "AHK", "AHL", "AHN", "AHX", "AI1", "AIA", "AII", "AIM", "AIQ", "AIS", "AIV", "AIZ",
    "AJL", "AJX", "AKA", "AKG", "AKM", "AKN", "AKO", "AKP", "AL3", "ALA", "ALB", "ALC", "ALD", "ALI", "ALK", "ALL", "ALM", "ALQ", "ALR", "ALV", "ALX", "ALY",
    "AM5", "AM7", "AMA", "AMC", "AMD", "AMH", "AMI", "AMN", "AMO", "AMP", "AMS", "AMU", "AMX", "AN1", "ANG", "ANN", "ANO", "ANR", "ANX", "ANZ", "AO1", "AOA", "AOF", "AOH", "AOK", "AON", "AOV", "APA", "APC", "APE", "APL", "APW", "APX", "APZ",
    "AQC", "AQD", "AQI", "AQN", "AQX", "AQZ", "AR1", "AR3", "AR9", "ARA", "ARB", "ARC", "ARD", "ARF", "ARG", "ARI", "ARL", "ARN", "ARR", "ART", "ARU", "ARV", "ARX",
    "AS1", "AS2", "ASB", "ASE", "ASG", "ASH", "ASK", "ASL", "ASM", "ASN", "ASP", "ASQ", "ASV", "ASX", "AT1", "ATA", "ATC", "ATG", "ATH", "ATM", "ATP", "ATR", "ATS", "ATT", "ATV", "ATX",
    "AU1", "AUA", "AUB", "AUC", "AUE", "AUG", "AUH", "AUI", "AUK", "AUN", "AUQ", "AUR", "AUV", "AUZ", "AV1", "AVA", "AVD", "AVE", "AVG", "AVH", "AVL", "AVM", "AVR", "AVW",
    "AW1", "AWJ", "AX1", "AX8", "AXE", "AXI", "AXL", "AXN", "AXP", "AYA", "AYM", "AYT", "AZ9", "AZI", "AZJ", "AZY",
    
    # B
    "B4P", "BAP", "BAS", "BB1", "BBC", "BBL", "BBN", "BBT", "BC8", "BCA", "BCB", "BCC", "BCI", "BCK", "BCM", "BCN", "BDG", "BDM", "BDX", "BEL", "BEN", "BEO", "BET", "BEZ", "BFG", "BFL", "BGA", "BGD", "BGE", "BGL", "BGP", "BGT", "BHD", "BHM", "BHP", "BIM", "BIO", "BIS", "BIT", "BKI", "BKT", "BKY", "BLG", "BLU", "BLX", "BLZ",
    "BM1", "BM8", "BMG", "BMH", "BML", "BMM", "BMN", "BMO", "BMR", "BMT", "BNL", "BNR", "BNZ", "BOA", "BOC", "BOD", "BOE", "BOL", "BOQ", "BOT", "BP8", "BPH", "BPM", "BPP", "BPT", "BRE", "BRG", "BRI", "BRK", "BRL", "BRN", "BRU", "BRX",
    "BSA", "BSL", "BSN", "BSX", "BTC", "BTE", "BTI", "BTL", "BTM", "BTN", "BTR", "BUB", "BUR", "BUS", "BUX", "BUY", "BVR", "BVS", "BWE", "BWF", "BWN", "BWP", "BXB", "BXN", "BYH",
    
    # C
    "C1X", "C29", "C79", "C7A", "CAA", "CAE", "CAEN", "CAF", "CAM", "CAN", "CAQ", "CAR", "CAT", "CAV", "CAY", "CAZ", "CBA", "CBE", "CBL", "CBO", "CBY", "CC5", "CC9", "CCA", "CCE", "CCG", "CCL", "CCM", "CCO", "CCP", "CCR", "CCV", "CCX", "CD1", "CD2", "CD3", "CDA", "CDE", "CDM", "CDO", "CDP", "CDR", "CDT", "CEH", "CEL", "CEN", "CF1", "CG1", "CGF", "CGO", "CGR", "CGS", "CHC", "CHL", "CHM", "CHN", "CHR", "CHRCA", "CHW", "CI1", "CIA", "CIN", "CIP", "CIW", "CKA", "CKF", "CL8", "CLA", "CLE", "CLG", "CLU", "CLV", "CLW", "CLX", "CLZ", "CMB", "CMD", "CMG", "CML", "CMM", "CMO", "CMP", "CMW", "CMX", "CNB", "CND", "CNI", "CNJ", "CNQ", "CNU", "COB", "COD", "COF", "COG", "COH", "COI", "COL", "COS", "COV", "COY", "CP8", "CPM", "CPN", "CPO", "CPU", "CPV", "CQE", "CQR", "CQT", "CR1", "CR3", "CR9", "CRB", "CRD", "CRI", "CRN", "CRR", "CRS", "CSC", "CSL", "CST", "CSX", "CT1", "CTD", "CTE", "CTM", "CTN", "CTO", "CTP", "CTQ", "CTT", "CU6", "CUE", "CUF", "CUL", "CUP", "CUV", "CVB", "CVC", "CVL", "CVN", "CVR", "CVV", "CVW", "CWP", "CWX", "CWY", "CXL", "CXO", "CXU", "CXZ", "CY5", "CYB", "CYC", "CYG", "CYL", "CYM", "CYMN", "CYP", "CYQ", "CZN", "CZR",
    
    # D
    "D2O", "D3E", "DAF", "DAI", "DAL", "DBF", "DBI", "DBO", "DCC", "DDR", "DDT", "DEL", "DEM", "DES", "DEV", "DGH", "DGL", "DGR", "DGT", "DJW", "DKM", "DLI", "DM1", "DME", "DMG", "DMM", "DMP", "DN1", "DNL", "DOC", "DOW", "DPM", "DRE", "DRO", "DRR", "DRX", "DSK", "DTI", "DTL", "DTM", "DTR", "DTZ", "DUB", "DUG", "DUI", "DUN", "DUR", "DVL", "DVP", "DWG", "DXB", "DXC", "DXI", "DXN", "DXS", "DY6", "DYL", "DYM",
    
    # E
    "E25", "E79", "EAT", "EAX", "EBO", "EBR", "ECF", "ECH", "ECL", "ECP", "ECS", "ECT", "EDC", "EDE", "EDU", "EDV", "EE1", "EEL", "EFE", "EG1", "EGG", "EGH", "EGL", "EGR", "EGY", "EHL", "EIQ", "EL8", "ELD", "ELS", "ELT", "ELV", "EM2", "EMB", "EMC", "EMD", "EME", "EMH", "EML", "EMN", "EMP", "EMR", "EMS", "EMT", "EMU", "EMUCA", "EMUND", "EMV", "ENL", "ENN", "ENR", "ENT", "ENV", "ENX", "EOL", "EOS", "EPM", "EPN", "EPX", "EPY", "EQN", "EQR", "EQS", "EQT", "EQX", "ERA", "ERD", "ERG", "ERL", "ERM", "ESK", "ESR", "ETM", "EUR", "EV1", "EV8", "EVE", "EVG", "EVN", "EVO", "EVR", "EVT", "EVZ", "EWC", "EXL", "EXP", "EXR", "EXT", "EYE", "EZL", "EZZ",
    
    # F
    "FAL", "FAR", "FAU", "FBM", "FBR", "FBU", "FCG", "FCL", "FCT", "FDR", "FDV", "FEG", "FEX", "FFG", "FFI", "FFM", "FG1", "FGG", "FGH", "FGR", "FGX", "FHE", "FHS", "FID", "FIN", "FL1", "FLC", "FLG", "FLN", "FLT", "FLX", "FME", "FMG", "FML", "FMR", "FND", "FNR", "FNX", "FOS", "FPC", "FPH", "FPR", "FRB", "FRE", "FRI", "FRM", "FRS", "FRW", "FRX", "FSA", "FSI", "FTI", "FUL", "FUN", "FWD", "FXG", "FZR",
    
    # G
    "G11", "G50", "G6M", "G88", "GA8", "GAL", "GAP", "GAS", "GBE", "GBR", "GBZ", "GC1", "GCI", "GCM", "GCR", "GDF", "GDG", "GDI", "GDM", "GED", "GEM", "GEN", "GES", "GFL", "GG8", "GGE", "GGP", "GHM", "GHY", "GIB", "GL1", "GLA", "GLB", "GLE", "GLF", "GLH", "GLL", "GLN", "GMD", "GMG", "GML", "GMN", "GNC", "GNE", "GNG", "GNM", "GNP", "GOR", "GOW", "GOZ", "GPR", "GPT", "GQG", "GR8", "GRE", "GRL", "GRR", "GRV", "GRX", "GSM", "GSN", "GSS", "GT1", "GT3", "GTE", "GTG", "GTH", "GTI", "GTK", "GTN", "GUE", "GUL", "GUM", "GVF", "GW1", "GWA", "GWR", "GYG",
    
    # H
    "H2G", "HAL", "HALN", "HAR", "HAS", "HAV", "HAW", "HCD", "HCF", "HCH", "HCL", "HCT", "HCW", "HDN", "HE8", "HFR", "HFY", "HGH", "HGO", "HHR", "HIO", "HIQ", "HIT", "HLI", "HLO", "HLS", "HLX", "HM1", "HMC", "HMD", "HMG", "HMI", "HMX", "HMY", "HNG", "HOR", "HPC", "HPG", "HPR", "HRE", "HRN", "HRZ", "HSN", "HT8", "HTG", "HTM", "HUB", "HUM", "HVN", "HVY", "HWK", "HXL", "HYD", "HYT", "HZN", "HZR",
    
    # I
    "I88", "IAG", "IAM", "IBC", "IBX", "ICE", "ICI", "ICL", "ICN", "ICR", "ICU", "ID8", "IDA", "IDT", "IDX", "IEL", "IEQ", "IFG", "IFL", "IFM", "IFN", "IFT", "IG6", "IGL", "IGN", "IGO", "IIQ", "IKE", "ILA", "ILT", "ILU", "IMA", "IMB", "IMC", "IMD", "IME", "IMI", "IMM", "IMR", "IMU", "INA", "IND", "INF", "ING", "INR", "INV", "IOD", "ION", "IPB", "IPC", "IPD", "IPG", "IPH", "IPT", "IPX", "IR1", "IRD", "IRE", "IRI", "IRX", "IS3", "ITM", "IVG", "IVR", "IVX", "IVZ", "IXC", "IXR",
    
    # J
    "JAL", "JAN", "JAT", "JAV", "JAY", "JBH", "JBY", "JCS", "JDO", "JGH", "JHX", "JIN", "JLG", "JLL", "JMS", "JNO", "JNS", "JPR", "JYC",
    
    # K
    "KAI", "KAL", "KAM", "KAR", "KAT", "KAU", "KBC", "KCC", "KCN", "KEY", "KFM", "KGD", "KGL", "KGN", "KKC", "KKO", "KLI", "KLR", "KLS", "KM1", "KMD", "KME", "KNB", "KNG", "KNI", "KNM", "KNO", "KOB", "KOR", "KOV", "KP2", "KPG", "KPO", "KRM", "KRR", "KSC", "KSL", "KSN", "KTA", "KYP", "KZR",
    
    # L
    "L1M", "LAM", "LAT", "LAU", "LBL", "LCE", "LCL", "LCY", "LDR", "LDX", "LEG", "LEL", "LEX", "LF1", "LFG", "LFS", "LGI", "LGL", "LGM", "LGP", "LIC", "LIN", "LIO", "LIS", "LIT", "LKE", "LKO", "LKY", "LLC", "LLM", "LM1", "LM8", "LMG", "LML", "LMS", "LNQ", "LNU", "LNW", "LOC", "LOT", "LOV", "LPE", "LPM", "LRD", "LRK", "LRM", "LRT", "LRV", "LSA", "LSF", "LSR", "LSX", "LTP", "LTR", "LU7", "LVE", "LYC", "LYL",
    
    # M
    "M24", "M2M", "M2R", "M3M", "M4M", "M79", "M7T", "MA1", "MA1NA", "MAC", "MAD", "MAF", "MAG", "MAH", "MAM", "MAN", "MAP", "MAQ", "MAT", "MAU", "MAUCA", "MAY", "MBH", "MBK", "MBX", "MC2", "MCA", "MCE", "MCM", "MCO", "MCP", "MCY", "MDI", "MDR", "MDX", "MEC", "MEG", "MEI", "MEK", "MEL", "MEM", "MEU", "MEZ", "MFD", "MFF", "MFG", "MGA", "MGH", "MGL", "MGR", "MGT", "MGU", "MGX", "MHC", "MHJ", "MHK", "MHM", "MI6", "MIN", "MIO", "MIR", "MKR", "MLG", "MLS", "MLX", "MM1", "MM8", "MMA", "MME", "MMI", "MML", "MMR", "MMS", "MNB", "MNC", "MND", "MOH", "MOM", "MOT", "MOV", "MP1", "MPA", "MPK", "MPL", "MPP", "MPR", "MPW", "MPX", "MQG", "MQR", "MRC", "MRD", "MRE", "MRI", "MRQ", "MRR", "MRZ", "MSB", "MSG", "MSI", "MSV", "MTB", "MTC", "MTH", "MTL", "MTM", "MTO", "MTS", "MVF", "MVL", "MVP", "MX1", "MXI", "MXO", "MXT", "MYE", "MYG", "MYR", "MYS", "MYX",
    
    # N
    "N1H", "NAB", "NAC", "NAE", "NAG", "NAN", "NC1", "NC6", "NCC", "NCK", "NDO", "NEC", "NEM", "NES", "NET", "NEU", "NFL", "NFM", "NGE", "NGI", "NGS", "NGX", "NGY", "NH3", "NHC", "NHE", "NHF", "NIC", "NIM", "NME", "NMG", "NMR", "NMT", "NNG", "NNL", "NOL", "NOR", "NOU", "NOV", "NOX", "NPM", "NRX", "NRZ", "NSB", "NSC", "NSM", "NSR", "NST", "NSX", "NTD", "NTI", "NTM", "NTU", "NUC", "NUF", "NUZ", "NVA", "NVO", "NVQ", "NVU", "NVX", "NWF", "NWH", "NWL", "NWM", "NWS", "NWSLV", "NXD", "NXG", "NXL", "NXM", "NXS", "NXT", "NYM", "NYR", "NZK", "NZM", "NZS",
    
    # O
    "OAK", "OB1DD", "OBL", "OBM", "OCA", "OCC", "OCL", "OCN", "OCT", "OD6", "ODA", "ODE", "ODY", "OEC", "OEL", "OEQ", "OFX", "OIL", "OKJ", "OLH", "OLI", "OLL", "OLY", "OM1", "OMA", "OMG", "OMH", "OML", "OMX", "ONE", "OPH", "OPN", "OPT", "ORA", "ORD", "ORE", "ORG", "ORI", "ORN", "ORP", "OSL", "OSM", "OSX", "OVT", "OZM", "OZZ",
    
    # P
    "PAB", "PAC", "PAR", "PAT", "PBH", "PBL", "PCG", "PCI", "PCK", "PCL", "PCX", "PDI", "PDN", "PE1", "PEB", "PEC", "PEK", "PEN", "PER", "PET", "PEX", "PFE", "PFG", "PFM", "PFP", "PFT", "PGC", "PGD", "PGF", "PGM", "PGO", "PGY", "PH2", "PHO", "PHX", "PIA", "PIC", "PIL", "PIM", "PIQ", "PKD", "PKO", "PKY", "PL3", "PL8", "PL9", "PLA", "PLC", "PLG", "PLN", "PLS", "PLT", "PLY", "PMC", "PME", "PMT", "PMV", "PNC", "PNI", "PNM", "PNN", "PNR", "PNT", "PNV", "POD", "POL", "PPC", "PPE", "PPG", "PPK", "PPL", "PPM", "PPS", "PPT", "PPY", "PR1", "PR2", "PRG", "PRM", "PRN", "PRO", "PRS", "PRSN", "PRU", "PRX", "PSC", "PSL", "PSQ", "PTL", "PTM", "PTN", "PTR", "PTX", "PUR", "PV1", "PVE", "PVL", "PVT", "PVW", "PWH", "PWN", "PWR", "PXA", "PXX", "PYC",
    
    # Q
    "QAL", "QAN", "QBE", "QEM", "QFE", "QGL", "QML", "QOR", "QPM", "QRI", "QUB", "QUE", "QXR",
    
    # R
    "R8R", "RAC", "RAD", "RAG", "RAN", "RAS", "RAU", "RB6", "RBD", "RBR", "RBX", "RC1", "RCE", "RCL", "RCM", "RCR", "RCT", "RDM", "RDN", "RDS", "RDX", "RDY", "REA", "REC", "REE", "REG", "REH", "REM", "REP", "REV", "REZ", "RF1", "RFF", "RFG", "RFT", "RG8", "RGL", "RGN", "RGT", "RHC", "RHI", "RHT", "RHY", "RIC", "RIE", "RIL", "RIM", "RIO", "RKB", "RKN", "RKT", "RLC", "RLF", "RLG", "RLT", "RMC", "RMD", "RMI", "RML", "RMS", "RMX", "RMY", "RND", "RNT", "RNU", "RNV", "RNX", "ROC", "ROG", "RON", "RPG", "RPL", "RPM", "RR1", "RRL", "RRR", "RSG", "RTG", "RTH", "RTR", "RUL", "RVT", "RWC", "RWD", "RWL", "RXH", "RXL", "RXR", "RYD", "RYZ", "RZI",
    
    # S
    "S2R", "S32", "S66", "SAN", "SB2", "SBM", "SBR", "SCG", "SCN", "SCP", "SCT", "SDF", "SDI", "SDR", "SDV", "SEC", "SEG", "SEK", "SEN", "SEQ", "SER", "SFC", "SFG", "SFM", "SFR", "SFX", "SGA", "SGH", "SGI", "SGLLV", "SGM", "SGP", "SGQ", "SGR", "SHA", "SHE", "SHG", "SHJ", "SHL", "SHM", "SHN", "SHO", "SHP", "SHV", "SIG", "SIO", "SIQ", "SIS", "SIV", "SIX", "SKC", "SKK", "SKM", "SKN", "SKO", "SKS", "SKT", "SKY", "SLA", "SLB", "SLC", "SLM", "SLS", "SLX", "SLZ", "SM1", "SMI", "SMM", "SMN", "SMP", "SMR", "SMS", "SMX", "SNC", "SND", "SNG", "SNL", "SNS", "SNT", "SNX", "SNZ", "SOC", "SOL", "SOM", "SOP", "SOR", "SP3", "SP8", "SPA", "SPD", "SPG", "SPK", "SPL", "SPN", "SPQ", "SPX", "SPZ", "SQX", "SRG", "SRH", "SRJ", "SRK", "SRL", "SRN", "SRR", "SRT", "SRV", "SRZ", "SS1", "SSG", "SSH", "SSL", "SSM", "SST", "ST1", "STG", "STH", "STK", "STM", "STN", "STO", "STP", "STX", "SUH", "SUL", "SUM", "SUN", "SVG", "SVL", "SVM", "SVR", "SVY", "SW1", "SWM", "SWP", "SX2", "SXE", "SXL", "SYL", "SYR",
    
    # T
    "T3D", "T88", "T92", "TAH", "TAL", "TAM", "TAR", "TAS", "TAT", "TBN", "TBR", "TCF", "TCG", "TCL", "TCO", "TD1", "TDO", "TEA", "TEE", "TEG", "TEK", "TEM", "TER", "TFL", "TG1", "TG6", "TGF", "TGH", "TGM", "TGN", "TGP", "THB", "THL", "THR", "TI1", "TIA", "TIG", "TIP", "TKL", "TKM", "TLC", "TLG", "TLM", "TLS", "TLX", "TM1", "TMB", "TMG", "TMK", "TML", "TMS", "TMX", "TNC", "TNE", "TNY", "TOE", "TOK", "TON", "TOP", "TOR", "TOT", "TOU", "TPC", "TPG", "TPW", "TR2", "TRA", "TRE", "TRI", "TRJ", "TRM", "TRP", "TRU", "TSI", "TSL", "TSO", "TTM", "TTT", "TTX", "TUA", "TVL", "TVN", "TWD", "TWE", "TWR", "TYP", "TYR", "TYX", "TZL", "TZN",
    
    # U
    "UBI", "UBN", "UCM", "UNI", "UNT", "UOS", "URF", "USL", "UVA", "UWC", "VAR", "VAU", "VBC", "VBS", "VBX", "VCX", "VEA", "VEE", "VEN", "VFX", "VFY", "VG1", "VGL", "VGN", "VHL", "VHM", "VIG", "VIT", "VKA", "VLS", "VMC", "VMG", "VML", "VMM", "VMT", "VN8", "VNL", "VNT", "VPR", "VR1", "VR8", "VRC", "VRL", "VRS", "VRX", "VSL", "VSR", "VTM", "VTX", "VUL", "VVA", "VYS",
    
    # W
    "W2V", "WA1", "WA8", "WAA", "WAF", "WAG", "WAK", "WAM", "WAR", "WAT", "WAX", "WBC", "WBE", "WBT", "WC1", "WC8", "WCE", "WCN", "WDS", "WEB", "WEC", "WEL", "WES", "WGB", "WGN", "WGR", "WGX", "WHC", "WHF", "WHI", "WHK", "WIA", "WIN", "WJL", "WLD", "WLE", "WMA", "WMG", "WMI", "WMX", "WNR", "WNX", "WOA", "WOR", "WOT", "WOW", "WPR", "WQG", "WR1", "WRK", "WSI", "WSR", "WTC", "WTL", "WTM", "WTN", "WWG", "WWI", "WYX", "WZR",
    
    # X
    "X2M", "XF1", "XGL", "XPN", "XRA", "XRF", "XRG", "XRO", "XST", "XTC", "XYZ",
    
    # Y
    "YAL", "YAR", "YOJ", "YOW", "YRL", "YUG",
    
    # Z
    "ZAG", "ZEO", "ZEU", "ZGL", "ZIM", "ZIP", "ZLD", "ZMI", "ZMM", "ZNC", "ZNO"
]

# Major ASX 200 companies (top tier)
ASX_200_SYMBOLS = [
    "ANZ", "BHP", "CBA", "CSL", "NAB", "WBC", "WES", "WOW", "TLS", "TCL",
    "RIO", "FMG", "BXB", "STO", "ORG", "S32", "WDS", "AGL", "COH", "COL",
    "REA", "SCG", "VCX", "NXT", "JBH", "HVN", "TWE", "A2M", "XRO", "NAN",
    "MQG", "AMP", "IAG", "SUN", "QBE", "ALL", "ASX", "BEN", "BOQ", "SGP",
    "GPT", "DMP", "GMG", "SGR", "TCL", "TLS", "WOW", "WES", "WBC", "WDS"
]

# Sector categorization
SECTOR_CATEGORIES = {
    "Banks": ["ANZ", "CBA", "NAB", "WBC", "BOQ", "BEN"],
    "Mining": ["BHP", "RIO", "FMG", "S32", "ORG", "STO", "WDS", "S32"],
    "Energy": ["WDS", "STO", "ORG", "S32"],
    "Healthcare": ["CSL", "COH", "COL", "RMD", "SUN", "TLS"],
    "Technology": ["XRO", "REA", "A2M", "NAN", "TLS", "TCL"],
    "REITs": ["SGP", "GPT", "DMP", "GMG", "SGR"],
    "Industrials": ["WES", "TCL", "TLS", "WOW", "HVN", "JBH"],
    "Consumer": ["WOW", "WES", "COH", "COL", "HVN", "JBH", "TWE"],
    "Telecom": ["TLS", "TCL"],
    "Financials": ["AMP", "IAG", "QBE", "ALL", "ASX", "MQG"],
    "Utilities": ["AGL", "ORG", "STO"]
}

# Market capitalization tiers
MARKET_CAP_TIERS = {
    "Large_Cap": ASX_200_SYMBOLS[:50],  # Top 50
    "Mid_Cap": ASX_200_SYMBOLS[50:],    # Rest of ASX 200
    "Small_Cap": [symbol for symbol in ASX_SYMBOLS if symbol not in ASX_200_SYMBOLS]
}

# Liquidity tiers (based on average daily volume)
LIQUIDITY_TIERS = {
    "High_Liquidity": ["BHP", "CBA", "CSL", "ANZ", "NAB", "WBC", "RIO", "WES", "WOW", "TLS"],
    "Medium_Liquidity": ASX_200_SYMBOLS[10:50],
    "Low_Liquidity": [symbol for symbol in ASX_SYMBOLS if symbol not in ASX_200_SYMBOLS]
}

def get_symbols_by_sector(sector: str) -> list:
    """Get symbols for a specific sector"""
    return SECTOR_CATEGORIES.get(sector, [])

def get_symbols_by_market_cap(tier: str) -> list:
    """Get symbols by market cap tier"""
    return MARKET_CAP_TIERS.get(tier, [])

def get_symbols_by_liquidity(tier: str) -> list:
    """Get symbols by liquidity tier"""
    return LIQUIDITY_TIERS.get(tier, [])

def get_all_symbols() -> list:
    """Get all ASX symbols"""
    return ASX_SYMBOLS.copy()

def get_asx_200_symbols() -> list:
    """Get ASX 200 symbols"""
    return ASX_200_SYMBOLS.copy()

def get_symbol_info(symbol: str) -> dict:
    """Get information about a specific symbol"""
    info = {
        "symbol": symbol,
        "is_asx_200": symbol in ASX_200_SYMBOLS,
        "sectors": [],
        "market_cap_tier": None,
        "liquidity_tier": None
    }
    
    # Find sectors
    for sector, symbols in SECTOR_CATEGORIES.items():
        if symbol in symbols:
            info["sectors"].append(sector)
    
    # Find market cap tier
    for tier, symbols in MARKET_CAP_TIERS.items():
        if symbol in symbols:
            info["market_cap_tier"] = tier
            break
    
    # Find liquidity tier
    for tier, symbols in LIQUIDITY_TIERS.items():
        if symbol in symbols:
            info["liquidity_tier"] = tier
            break
    
    return info