"""
Comprehensive Test Suite for Phase 1 Dataset Audit Edge Cases.
Validates all 11 required edge cases using temporary CSV files.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
from car_price_prediction import run_phase1_audit

def run_all_tests():
    print("=" * 72)
    print(" STARTING TEST SUITE FOR 11 EDGE-CASE SCENARIOS")
    print("=" * 72)
    
    results = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Baseline valid CSV row data
        header_valid = "Car_Name,Year,Selling_Price,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner\n"
        row1 = "city,2015,5.5,9.5,30000,Petrol,Dealer,Manual,0\n"
        row2 = "brio,2014,4.2,6.8,25000,Petrol,Dealer,Manual,0\n"
        row3 = "verna,2016,6.8,11.2,35000,Diesel,Dealer,Manual,0\n"
        valid_csv_content = header_valid + row1 + row2 + row3

        # -------------------------------------------------------------
        # Test 1: Valid expected schema
        # -------------------------------------------------------------
        t1_file = tmp_path / "test1_valid.csv"
        t1_file.write_text(valid_csv_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 1: Valid expected schema")
        status1 = run_phase1_audit(t1_file)
        results["Test 1: Valid expected schema"] = (status1 in ("PASS", "PASS WITH WARNINGS"), f"Status={status1}")

        # -------------------------------------------------------------
        # Test 2: Duplicate raw column headers
        # -------------------------------------------------------------
        t2_file = tmp_path / "test2_dup_headers.csv"
        t2_content = "Car_Name,Year,Selling_Price,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner,Year\n" + row1
        t2_file.write_text(t2_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 2: Duplicate raw column headers")
        status2 = run_phase1_audit(t2_file)
        results["Test 2: Duplicate raw column headers"] = (status2 == "FAIL", f"Status={status2}")

        # -------------------------------------------------------------
        # Test 3: Headers duplicated after trimming whitespace or ignoring case
        # -------------------------------------------------------------
        t3_file = tmp_path / "test3_norm_dup_headers.csv"
        t3_content = "Car_Name,Year,Selling_Price,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner, Year \n" + row1
        t3_file.write_text(t3_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 3: Normalized duplicate headers ('Year' vs ' Year ')")
        status3 = run_phase1_audit(t3_file)
        results["Test 3: Normalized duplicate headers"] = (status3 == "FAIL", f"Status={status3}")

        # -------------------------------------------------------------
        # Test 4: Blank headers
        # -------------------------------------------------------------
        t4_file = tmp_path / "test4_blank_headers.csv"
        t4_content = "Car_Name,,Selling_Price,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner\n" + row1
        t4_file.write_text(t4_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 4: Blank column header")
        status4 = run_phase1_audit(t4_file)
        results["Test 4: Blank column header"] = (status4 == "FAIL", f"Status={status4}")

        # -------------------------------------------------------------
        # Test 5: Missing required columns
        # -------------------------------------------------------------
        t5_file = tmp_path / "test5_missing_req.csv"
        t5_content = "Car_Name,Year,Selling_Price,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission\n" + "city,2015,5.5,9.5,30000,Petrol,Dealer,Manual\n"
        t5_file.write_text(t5_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 5: Missing required column ('Owner' missing)")
        status5 = run_phase1_audit(t5_file)
        results["Test 5: Missing required columns"] = (status5 == "FAIL", f"Status={status5}")

        # -------------------------------------------------------------
        # Test 6: Unexpected extra columns
        # -------------------------------------------------------------
        t6_file = tmp_path / "test6_extra_col.csv"
        t6_content = "Car_Name,Year,Selling_Price,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner,Color\n" + "city,2015,5.5,9.5,30000,Petrol,Dealer,Manual,0,Red\n" + "brio,2014,4.2,6.8,25000,Petrol,Dealer,Manual,0,Blue\n" + "verna,2016,6.8,11.2,35000,Diesel,Dealer,Manual,0,White\n"
        t6_file.write_text(t6_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 6: Unexpected extra column ('Color')")
        status6 = run_phase1_audit(t6_file)
        results["Test 6: Unexpected extra columns"] = (status6 == "PASS WITH WARNINGS", f"Status={status6}")

        # -------------------------------------------------------------
        # Test 7: Empty cells and pseudo-missing tokens
        # -------------------------------------------------------------
        t7_file = tmp_path / "test7_missing_tokens.csv"
        # Fuel_Type row 1 has empty string (Category A), row 2 has 'null', row 3 has '?' (Category B)
        t7_content = header_valid + \
            "city,2015,5.5,9.5,30000,,Dealer,Manual,0\n" + \
            "brio,2014,4.2,6.8,25000,null,Dealer,Manual,0\n" + \
            "verna,2016,6.8,11.2,35000,?,Dealer,Manual,0\n"
        t7_file.write_text(t7_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 7: Empty cells and pseudo-missing tokens")
        status7 = run_phase1_audit(t7_file)
        results["Test 7: Empty cells and pseudo-missing tokens"] = (status7 == "PASS WITH WARNINGS", f"Status={status7}")

        # -------------------------------------------------------------
        # Test 8: Non-numeric strings in numeric columns
        # -------------------------------------------------------------
        t8_file = tmp_path / "test8_non_numeric.csv"
        t8_content = header_valid + "city,2015,INVALID_PRICE,9.5,30000,Petrol,Dealer,Manual,0\n"
        t8_file.write_text(t8_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 8: Non-numeric strings in numeric columns")
        status8 = run_phase1_audit(t8_file)
        results["Test 8: Non-numeric strings in numeric columns"] = (status8 == "FAIL", f"Status={status8}")

        # -------------------------------------------------------------
        # Test 9: Infinite numeric values
        # -------------------------------------------------------------
        t9_file = tmp_path / "test9_inf_values.csv"
        t9_content = header_valid + \
            "city,2015,5.5,inf,30000,Petrol,Dealer,Manual,0\n" + \
            "brio,2014,4.2,6.8,25000,Petrol,Dealer,Manual,0\n" + \
            "verna,2016,6.8,11.2,35000,Diesel,Dealer,Manual,0\n"
        t9_file.write_text(t9_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 9: Infinite numeric values")
        status9 = run_phase1_audit(t9_file)
        results["Test 9: Infinite numeric values"] = (status9 == "PASS WITH WARNINGS", f"Status={status9}")

        # -------------------------------------------------------------
        # Test 10: Missing Car_Name or Year during business-logic validation
        # -------------------------------------------------------------
        t10_file = tmp_path / "test10_missing_optional_in_logic.csv"
        # Omit Car_Name and Year, but have Selling_Price > Present_Price (inversion)
        t10_content = "Selling_Price,Present_Price,Driven_kms,Fuel_Type,Selling_type,Transmission,Owner\n" + \
            "15.0,10.0,30000,Petrol,Dealer,Manual,0\n"
        t10_file.write_text(t10_content, encoding="utf-8")
        print("\n>>> RUNNING TEST 10: Missing Car_Name and Year during business logic")
        status10 = run_phase1_audit(t10_file)
        # Should be FAIL due to missing required schema columns, but NOT crash with KeyError in business logic!
        results["Test 10: Missing Car_Name/Year during business logic"] = (status10 == "FAIL", f"Status={status10}")

        # -------------------------------------------------------------
        # Test 11: Empty or malformed CSV files
        # -------------------------------------------------------------
        t11_file = tmp_path / "test11_empty.csv"
        t11_file.write_text("", encoding="utf-8")  # 0-byte file
        print("\n>>> RUNNING TEST 11: 0-byte empty CSV file")
        status11 = run_phase1_audit(t11_file)
        results["Test 11: Empty 0-byte CSV file"] = (status11 == "FAIL", f"Status={status11}")

    print("\n" + "=" * 72)
    print(" TEST SUITE SUMMARY RESULTS")
    print("=" * 72)
    all_passed = True
    for test_name, (passed, detail) in results.items():
        outcome = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  [{outcome}] {test_name:<55} -> {detail}")
    
    print("=" * 72)
    if all_passed:
        print(" ALL 11 EDGE-CASE TESTS PASSED SUCCESSFULLY!")
    else:
        print(" SOME TESTS FAILED!")
    print("=" * 72)
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
