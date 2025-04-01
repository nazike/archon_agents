from supabase import Client
from typing import List, Dict, Any, Optional
import os

def create_supabase_client() -> Client:
    """
    Create a Supabase client from environment variables
    
    Returns:
        Supabase client
    """
    from supabase import create_client
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
    return create_client(url, key)

async def upsert_document(
    supabase: Client,
    table_name: str,
    document: Dict[str, Any],
    id_field: str = "id"
) -> Dict[str, Any]:
    """
    Insert or update a document in Supabase
    
    Args:
        supabase: Supabase client
        table_name: Name of the table
        document: Document to upsert
        id_field: Name of the ID field
    
    Returns:
        Result of the operation
    """
    try:
        if id_field in document:
            # Update
            result = supabase.table(table_name).update(document).eq(id_field, document[id_field]).execute()
        else:
            # Insert
            result = supabase.table(table_name).insert(document).execute()
        
        return result.data[0] if result.data else {}
    except Exception as e:
        print(f"Error upserting document: {e}")
        return {}

async def batch_insert(
    supabase: Client,
    table_name: str,
    documents: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Insert documents in batches
    
    Args:
        supabase: Supabase client
        table_name: Name of the table
        documents: Documents to insert
        batch_size: Size of each batch
    
    Returns:
        List of results
    """
    results = []
    
    # Process in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            result = supabase.table(table_name).insert(batch).execute()
            if result.data:
                results.extend(result.data)
            print(f"Inserted batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error inserting batch: {e}")
    
    return results

async def list_unique_fields(
    supabase: Client,
    table_name: str,
    field_name: str,
    filter_condition: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    List unique values for a field
    
    Args:
        supabase: Supabase client
        table_name: Name of the table
        field_name: Name of the field
        filter_condition: Optional filter condition
    
    Returns:
        List of unique values
    """
    try:
        query = supabase.from_(table_name).select(field_name)
        
        # Apply filter if provided
        if filter_condition:
            for key, value in filter_condition.items():
                query = query.eq(key, value)
                
        result = query.execute()
        
        if not result.data:
            return []
            
        # Extract unique values
        values = set(item[field_name] for item in result.data if field_name in item)
        return sorted(values)
    except Exception as e:
        print(f"Error listing unique fields: {e}")
        return [] 