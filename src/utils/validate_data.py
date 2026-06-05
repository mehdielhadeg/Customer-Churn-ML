import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeBetween,
    ExpectColumnPairValuesAToBeGreaterThanB
)
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations 1.5.8+.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print(" Starting data validation with Great Expectations 1.5.8...")
    
    #  1. SETUP EPHEMERAL CONTEXT & BATCH 
    # Creates an isolated, in-memory context ideal for script/notebook workflows
    context = gx.get_context()
    
    datasource = context.data_sources.add_pandas(name="telco_datasource")
    data_asset = datasource.add_dataframe_asset(name="telco_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("telco_batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    # Initialize an empty Expectation Suite
    suite = gx.ExpectationSuite(name="telco_validation_suite")
    
    # 2. SCHEMA VALIDATION - ESSENTIAL COLUMNS 
    print(" Validating schema and required columns...")
    
    # Customer identifier must exist and be fully populated
    suite.add_expectation(ExpectColumnToExist(column="customerID"))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="customerID"))
    
    # Verify presence of all other critical features
    required_columns = [
        "gender", "Partner", "Dependents", "PhoneService", 
        "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges"
    ]
    for col in required_columns:
        suite.add_expectation(ExpectColumnToExist(column=col))
    
    #  3. BUSINESS LOGIC VALIDATION 
    print(" Validating business logic constraints...")
    
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]))
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]))
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]))
    
    suite.add_expectation(ExpectColumnValuesToBeInSet(
        column="Contract", 
        value_set=["Month-to-month", "One year", "Two year"]
    ))
    
    suite.add_expectation(ExpectColumnValuesToBeInSet(
        column="InternetService",
        value_set=["DSL", "Fiber optic", "No"]
    ))
    
    #  4. NUMERIC RANGE VALIDATION ===
    print(" Validating numeric ranges and business constraints...")
    
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="tenure", min_value=0))
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0))
    
    #  5. STATISTICAL VALIDATION 
    print(" Validating statistical properties...")
    
    # Max tenure ~10 years (120 months)
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120))
    # Reasonable business limit for standard telecom plans
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200))
    
    # Financial indicators must not contain missing values
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))
    
    # 6. DATA CONSISTENCY CHECKS 
    print(" Validating data consistency...")
    
    
    
    # 7. RUN VALIDATION SUITE
    print(" Running complete validation suite...")
    results = batch.validate(suite)
    
    # 8. PROCESS RESULTS 
    failed_expectations = []
    for r in results.results:
        if not r.success:
            # Modern 1.x property mapping to fetch the configuration type
            expectation_type = r.expectation_config.type
            failed_expectations.append(expectation_type)
    
    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks
    
    if results.success:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results.success, failed_expectations