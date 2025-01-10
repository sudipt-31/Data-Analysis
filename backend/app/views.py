# backend/app/views.py
from venv import logger
from rest_framework.views import APIView
from django.conf import settings


from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings

from langchain.prompts import PromptTemplate
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from dotenv import load_dotenv
import re
import io
from sqlalchemy import create_engine, text
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from io import StringIO
from django.http import HttpResponse


logger = logging.getLogger(__name__)




def get_api_key():
    """Get Google API key from Django settings or environment"""
    try:
        # First try Django settings
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise AttributeError
        return api_key
    except AttributeError:
        # Then try environment variable
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("Google API key not found. Please set GOOGLE_API_KEY in settings.py or .env file")
        return api_key

def setup_llm():
    """Initialize Gemini and chain with dynamic schema"""
    try:
        api_key = get_api_key()
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        prompt = get_prompt_template()
        return prompt, llm
    except Exception as e:
        raise Exception(f"Error setting up LLM: {str(e)}")

def get_prompt_template():
    """Get the prompt template with dynamic schema"""
    template = """
You are an expert SQL query generator for a dynamic database. Given a natural language question, generate the appropriate SQL query based on the following schema:

DATABASE SCHEMA:
{schema}

IMPORTANT RULES FOR SQL QUERY GENERATION:
1. Return ONLY the SQL query without explanations or comments.
2. Use appropriate JOIN clauses for combining tables.
3. If tables have columns with the same name, use aliases for each table.
4. If two columns are identical across tables, merge them into a single column by selecting only one.
5. Use relevant WHERE clauses for filtering and specify join conditions clearly.
6. Include aggregation functions (COUNT, AVG, SUM) when required.
7. Use GROUP BY for aggregated results and ORDER BY for sorting when applicable.
8. Select only needed columns instead of using *.
9. Always limit results to 10 rows unless asked otherwise.
10. Clearly assign aliases to each table and reference all columns with table aliases.
11. Always check the table schema and column names to ensure correct references.
12. For age calculations use: CAST(CAST(JULIANDAY(CURRENT_TIMESTAMP) - JULIANDAY(CAST(birth_year AS TEXT) || '-01-01') AS INT) / 365 AS INT)
13. Ensure foreign key relationships are correctly used in JOINs.
14. Use aggregation functions with actual columns from the correct table.
15. Use table aliases, but **do not use the 'AS' keyword for aliases** (e.g., `uploaded_data ud` instead of `uploaded_data AS ud`).
16. The SQL query should be able to run in SQLite.
17. most Don't use this format  ```sql ```  only give me the sql query.
18. IMPORTANT: When referencing column names that contain spaces or special characters, always wrap them in double quotes ("). For example: "Hair serums", "Product category"
19. For column names with spaces, use double quotes like this: SELECT "Hair serums" FROM table_name

User Question: {question}

Generate the SQL query that answers this question:
"""
    return PromptTemplate(
        input_variables=["question", "schema"],
        template=template
    )


def generate_result_explanation(results_df, user_question, llm):
    """Generate a clear, concise explanation of query results with key insights."""
    try:
        # Basic dataset info
        row_count = len(results_df)
        if row_count == 0:
            return "###No Results Found\nThe query returned no data. Please try modifying your search criteria."

        # Analyze numeric and categorical columns
        insights = []
        for column in results_df.columns:
            col_data = results_df[column]
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Only calculate stats if there's non-null numeric data
                if not col_data.isna().all():
                    stats = {
                        'mean': col_data.mean(),
                        'max': col_data.max(),
                        'min': col_data.min()
                    }
                    insights.append(f"- {column}: Range {stats['min']:.2f} to {stats['max']:.2f}, Average {stats['mean']:.2f}")
            else:
                # For categorical columns, show top values and their counts
                value_counts = col_data.value_counts().head(3)
                if not value_counts.empty:
                    top_values = ", ".join(f"{val} ({count})" for val, count in value_counts.items())
                    insights.append(f"- {column}: Most common values: {top_values}")

        # Create analysis prompt
        analysis_prompt = f"""
        Analyze this data summary for the question: "{user_question}"
        
        Dataset Overview:
        - Total records: {row_count}
        - Column insights:
        {chr(10).join(insights)}
        
        First few rows:
        {results_df.head(2).to_string()}
        
        Provide a 2-3 sentence summary that:
        1. Directly answers the user's question
        2. Highlights the most significant findings
        3. Mentions any notable patterns or trends
        """

        # Get explanation from LLM
        response = llm.invoke(analysis_prompt)
        explanation = response.content if hasattr(response, 'content') else str(response)
        
        # Format the final output
        formatted_explanation = f"""

{explanation}

- Records analyzed: {row_count:,}
- Columns analyzed: {len(results_df.columns)}"""
        
        return formatted_explanation

    except Exception as e:
        return f"""
        ### ⚠️ Analysis Error
        
        Unable to analyze results: {str(e)}
        
        Basic Information:
        - Records: {len(results_df)}
        - Columns: {', '.join(results_df.columns)}
        """


def clean_column_names(headers):
    
    cleaned_headers = []
    seen_headers = {}
    
    for header in headers:
        # Convert to string and clean
        if pd.isna(header) or str(header).strip() == '':
            header = "Unnamed_Column"
        else:
            # Keep the original header text but clean it for SQL compatibility
            header = str(header).strip()
            # Replace special characters except spaces
            header = re.sub(r'[^\w\s]', '_', header)
            # Replace multiple spaces with single underscore
            header = re.sub(r'\s+', '_', header)
            
        # Handle duplicate headers
        base_header = header
        counter = 1
        while header in seen_headers:
            header = f"{base_header}_{counter}"
            counter += 1
        
        seen_headers[header] = True
        cleaned_headers.append(header)
    
    return cleaned_headers

def restructure_excel_sheet(uploaded_file):
    """Restructure and clean Excel sheet data with enhanced table detection."""
    try:
        file_bytes = uploaded_file.read()
        excel_bytes = io.BytesIO(file_bytes)
        
        cleaned_dfs = {}
        
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(excel_bytes)
            sheets = excel_file.sheet_names
            
            for sheet in sheets:
                df = pd.read_excel(excel_bytes, sheet_name=sheet, header=None)
                if df.empty:
                    continue
                
                clean_sheet_name = re.sub(r'[^\w\s]', '_', sheet)
                clean_sheet_name = re.sub(r'\s+', '_', clean_sheet_name)
                
                table_sections = find_tables_in_dataframe(df, sheet_name=clean_sheet_name)
                
                for table_info in table_sections:
                    processed_df = process_single_table(df, table_info)
                    if processed_df is not None and not processed_df.empty:
                        table_name = table_info['name'].lower()
                        print(f"Processing table: {table_name} with {len(processed_df)} rows and {len(processed_df.columns)} columns")
                        cleaned_dfs[table_name] = processed_df
                        
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file_bytes.decode('utf-8')), header=None)
            if not df.empty:
                csv_name = os.path.splitext(uploaded_file.name)[0]
                clean_csv_name = re.sub(r'[^\w\s]', '_', csv_name)
                clean_csv_name = re.sub(r'\s+', '_', clean_csv_name)
                
                table_sections = find_tables_in_dataframe(df, sheet_name=clean_csv_name)
                
                for table_info in table_sections:
                    processed_df = process_single_table(df, table_info)
                    if processed_df is not None and not processed_df.empty:
                        table_name = table_info['name'].lower()
                        print(f"Processing table: {table_name} with {len(processed_df)} rows and {len(processed_df.columns)} columns")
                        cleaned_dfs[table_name] = processed_df
        
        return cleaned_dfs if cleaned_dfs else None
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
    finally:
        uploaded_file.seek(0)


def generate_and_execute_query(user_question, schema_str, llm, db_uri):
    """
    Generate and execute SQL query for PostgreSQL with improved case-sensitive column handling
    """
    try:
        # Get actual column names and their case from the database
        engine = create_engine(db_uri)
        with engine.connect() as connection:
            # Get all table names first
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            tables_result = connection.execute(text(tables_query))
            tables = [row[0] for row in tables_result]
            
            # Get column information for each table
            columns_info = {}
            for table in tables:
                columns_query = f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table}'
                """
                columns_result = connection.execute(text(columns_query))
                columns_info[table] = [row[0] for row in columns_result]

        # Add column case information to the schema string
        schema_str += "\nActual column names and their case:\n"
        for table, columns in columns_info.items():
            schema_str += f"\nTable '{table}':\n"
            for col in columns:
                schema_str += f"- {col}\n"

        # Generate initial query with explicit column case handling
        response = llm.invoke(f"""
        Generate a PostgreSQL-compatible query following these strict rules:
        1. Use EXACT column names as shown in the schema (case-sensitive)
        2. Always quote column names with double quotes
        3. For string comparisons, use ILIKE for case-insensitive matching
        4. Cast numeric values explicitly using CAST with correct column name case
        5. Use table name prefix for all columns
        
        Schema with exact column names:
        {schema_str}
        
        Question: {user_question}
        
        Return only the SQL query without any explanation or markdown formatting.
        """)
        
        sql_query = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up the query
        sql_query = sql_query.strip()
        if sql_query.startswith('```sql'):
            sql_query = sql_query[6:-3]
        sql_query = sql_query.strip()
        
        # Execute query
        try:
            with engine.connect() as connection:
                # Test query first
                test_query = text("EXPLAIN " + sql_query)
                connection.execute(test_query)
                
                # Execute actual query
                query = text(sql_query)
                result = connection.execute(query)
                results = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if results.empty:
                    # Try to debug with exact column names
                    debug_response = llm.invoke(f"""
                    The query returned no results. Modify it using exact column names:
                    Original query: {sql_query}
                    
                    Schema with exact column names:
                    {schema_str}
                    
                    Question: {user_question}
                    
                    Return only the fixed SQL query.
                    """)
                    
                    modified_query = debug_response.content.strip()
                    if modified_query.startswith('```sql'):
                        modified_query = modified_query[6:-3].strip()
                    
                    # Try modified query
                    query = text(modified_query)
                    result = connection.execute(query)
                    results = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    if not results.empty:
                        sql_query = modified_query
                
                return {
                    'success': True,
                    'query': sql_query,
                    'results': results
                }
                
        except Exception as e:
            # Attempt to fix the query with correct column names
            fix_response = llm.invoke(f"""
            Fix this query using exact column names from the schema.
            Error: {str(e)}
            Original query: {sql_query}
            
            Schema with exact column names:
            {schema_str}
            
            Return only the fixed SQL query.
            """)
            
            fixed_query = fix_response.content.strip()
            if fixed_query.startswith('```sql'):
                fixed_query = fixed_query[6:-3].strip()
            
            # Try the fixed query
            with engine.connect() as connection:
                query = text(fixed_query)
                result = connection.execute(query)
                results = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                return {
                    'success': True,
                    'query': fixed_query,
                    'results': results
                }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        if 'engine' in locals():
            engine.dispose()

def clear_database_tables(engine):
    """Clear all tables from the data_analysis database"""
    try:
        with engine.connect() as conn:
            # Disable foreign key checks and transactions
            conn.execute(text("SET session_replication_role = 'replica';"))
            
            # Get list of all tables in public schema
            table_names_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE';
            """
            result = conn.execute(text(table_names_query))
            tables = [row[0] for row in result]
            
            # Drop each table
            if tables:
                for table in tables:
                    conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE;'))
                print("Database 'data_analysis' cleared successfully - all existing tables removed.")
            else:
                print("Database 'data_analysis' is already empty.")
                
            # Re-enable foreign key checks
            conn.execute(text("SET session_replication_role = 'origin';"))
            conn.commit()
            
    except Exception as e:
        print(f"Error clearing database: {str(e)}")
        raise e

def is_table_name_row(row):
    """Check if a row contains only one non-empty cell that could be a table name."""
    non_empty_values = row.dropna()
    return len(non_empty_values) == 1

def is_header_row(row):
    """Check if a row could be a header row."""
    non_empty_values = row.dropna()
    return len(non_empty_values) > 1

def process_single_table(df, table_info):
    """Process a single table using the provided table information."""
    try:
        headers = df.iloc[table_info['header_row']].tolist()
        cleaned_headers = clean_column_names(headers)
        
        start_idx = table_info['data_start']
        end_idx = table_info['end']
        data_df = df.iloc[start_idx:end_idx].copy()
        
        result_df = pd.DataFrame(data_df.values, columns=cleaned_headers)
        result_df = result_df.dropna(how='all').dropna(axis=1, how='all')
        
        return result_df if not result_df.empty else None
        
    except Exception as e:
        print(f"Error processing table section: {str(e)}")
        return None

def find_tables_in_dataframe(df, sheet_name="default"):
    """Find multiple tables in a DataFrame with enhanced detection logic."""
    tables = []
    current_table = None
    i = 0
    table_counter = 1
    
    while i < len(df):
        row = df.iloc[i]
        
        if row.isna().all():
            if current_table is not None:
                tables.append(current_table)
                current_table = None
            i += 1
            continue
        
        if is_table_name_row(row):
            original_table_name = str(row.dropna().iloc[0]).strip()
            if original_table_name and isinstance(original_table_name, str):
                table_name = f"{sheet_name}_{original_table_name}"
            else:
                table_name = f"{sheet_name}_table_{table_counter}"
                table_counter += 1
            
            table_name = re.sub(r'[^\w\s]', '_', table_name)
            table_name = re.sub(r'\s+', '_', table_name)
            
            header_idx = None
            data_start_idx = None
            
            for j in range(i + 1, len(df)):
                next_row = df.iloc[j]
                if next_row.isna().all():
                    break
                if header_idx is None and is_header_row(next_row):
                    header_idx = j
                    data_start_idx = j + 1
                    break
            
            if header_idx is not None:
                if current_table is not None:
                    tables.append(current_table)
                
                current_table = {
                    'name': table_name,
                    'start': header_idx,
                    'data_start': data_start_idx,
                    'header_row': header_idx,
                    'end': None
                }
                i = data_start_idx
                continue
        
        if current_table is None and is_header_row(row):
            table_name = f"{sheet_name}_table_{table_counter}"
            table_name = re.sub(r'[^\w\s]', '_', table_name)
            table_name = re.sub(r'\s+', '_', table_name)
            
            current_table = {
                'name': table_name,
                'start': i,
                'data_start': i + 1,
                'header_row': i,
                'end': None
            }
            table_counter += 1
            i += 1
            continue
        
        if current_table is not None:
            current_table['end'] = i + 1
        
        i += 1
    
    if current_table is not None:
        tables.append(current_table)
    
    return tables



class DataAnalysisAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def __init__(self):
        super().__init__()
        self.llm = None
        self.prompt_template = None
        try:
            self.db_uri = settings.DATABASE_URL
        except AttributeError:
            raise Exception("DATABASE_URL not found in Django settings")
        self.initialize_llm()

    def initialize_llm(self):
        try:
            self.prompt_template, self.llm = setup_llm()
            if self.llm is None:
                raise Exception("Failed to initialize LLM")
        except Exception as e:
            raise Exception(f"Error initializing LLM: {str(e)}")



    def post(self, request, *args, **kwargs):
        """Handle both file uploads and analysis queries"""
        content_type = request.content_type if hasattr(request, 'content_type') else ''
        
        if content_type and 'multipart/form-data' in content_type:
            return self.handle_file_upload(request)
        elif content_type and 'application/json' in content_type:
            return self.handle_analysis_query(request)
        else:
            return Response({
                'error': f'Unsupported content type: {content_type}'
            }, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)


    def handle_file_upload(self, request):
        try:
            uploaded_file = request.FILES['file']
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension not in ['.xlsx', '.xls', '.csv']:
                return Response({
                    'error': 'Invalid file type. Please upload .xlsx, .xls, or .csv files.'
                }, status=status.HTTP_400_BAD_REQUEST)

            processed_data = restructure_excel_sheet(uploaded_file)
            
            if processed_data is None:
                return Response({
                    'error': 'No valid data found in the uploaded file.'
                }, status=status.HTTP_400_BAD_REQUEST)

            engine = create_engine(self.db_uri)
            clear_database_tables(engine)

            if isinstance(processed_data, dict):
                tables_created = []
                for sheet_name, df in processed_data.items():
                    table_name = re.sub(r'[^\w]', '_', sheet_name.lower())
                    df.to_sql(table_name, engine, if_exists='replace', index=False)
                    tables_created.append(table_name)

                return Response({
                    'success': True,
                    'message': 'Multiple tables created successfully',
                    'tables': tables_created
                })
            else:
                table_name = 'main_data'
                processed_data.to_sql(table_name, engine, if_exists='replace', index=False)
                return Response({
                    'success': True,
                    'message': 'Table created successfully',
                    'table': table_name
                })

        except Exception as e:
            return Response({
                'error': f'Error processing file: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            if 'engine' in locals():
                engine.dispose()

    def handle_analysis_query(self, request):
        try:
            user_question = request.data.get('query')
            if not user_question:
                return Response({
                    'error': 'Please provide a question.'
                }, status=status.HTTP_400_BAD_REQUEST)

            engine = create_engine(self.db_uri)
            schema_str = self.get_schema_info(engine)

            result = generate_and_execute_query(
                user_question,
                schema_str,
                self.llm,
                self.db_uri
            )

            if result['success']:
                if not result['results'].empty:
                    results_dict = result['results'].to_dict(orient='records')
                    explanation = generate_result_explanation(
                        result['results'],
                        user_question,
                        self.llm
                    )

                    return Response({
                        'success': True,
                        'query': result['query'],
                        'results': results_dict,
                        'explanation': explanation
                    })
                else:
                    return Response({
                        'success': True,
                        'message': 'No results found',
                        'query': result['query']
                    })
            else:
                return Response({
                    'error': result['error']
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            return Response({
                'error': f'Analysis error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            if 'engine' in locals():
                engine.dispose()

    def get_schema_info(self, engine):
    
        try:
            with engine.connect() as conn:
                # Get all tables
                tables_query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                        AND table_name != ''  
                        AND EXISTS (
                            SELECT 1 
                            FROM information_schema.columns 
                            WHERE table_name = tables.table_name
                        )
                """
                tables = pd.read_sql_query(tables_query, conn)
                
                schema_str = "Available tables and their columns:\n\n"
                for table_name in tables['table_name']:
                    # Get column information
                    columns_query = f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}'
                    """
                    columns = pd.read_sql_query(columns_query, conn)
                    
                    # Get sample data
                    sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                    sample_rows = pd.read_sql_query(sample_query, conn)
                    
                    schema_str += f"Table: {table_name}\n"
                    schema_str += "Columns:\n"
                    for _, col in columns.iterrows():
                        col_name = col['column_name']
                        col_type = col['data_type']
                        sample_vals = sample_rows[col_name].tolist() if not sample_rows.empty else ['NULL']
                        schema_str += f"- {col_name} ({col_type}) - Samples: {', '.join(str(v) for v in sample_vals)}\n"
                    schema_str += "\n"
                
                return schema_str
                
        except Exception as e:
            raise Exception(f"Error getting schema info: {str(e)}")

    def get_db_uri(self):
        """Get database URI from Django settings""" 
        try:
            return settings.DATABASE_URL
        except AttributeError:
            raise Exception("DATABASE_URL not found in settings")
    
    def cleanup_temporary_files(self):
        """Clean up temporary files after processing"""
        try:
            # Add cleanup logic
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    
    
class SaveResultsAPIView(APIView):
    def post(self, request):
        try:
            results_data = request.data.get('results', [])
            if not results_data:
                return Response({
                    'error': 'No results to save'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results_data)
            
            # Create a string buffer to store the CSV data
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            # Create the HTTP response with the CSV file
            filename = f'query_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
            response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            
            return response
            
        except Exception as e:
            return Response({
                'error': f'Error saving results: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
   
       