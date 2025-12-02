import pytest
import pandas as pd
import os
from src.parse import parse_movie_titles, parse_probe

def test_parse_movie_titles(tmp_path):
    # Create a dummy movie_titles.csv
    csv_content = "1,2003,Dinosaur Planet\n2,2004,Isle of Man TT 2004 Review\n3,NULL,Character\n"
    csv_file = tmp_path / "movie_titles.csv"
    csv_file.write_text(csv_content, encoding='ISO-8859-1')
    
    output_file = tmp_path / "movies.parquet"
    
    parse_movie_titles(str(csv_file), str(output_file))
    
    assert output_file.exists()
    df = pd.read_parquet(output_file)
    assert len(df) == 3
    assert df.iloc[0]['title'] == 'Dinosaur Planet'
    assert pd.isna(df.iloc[2]['year'])

def test_parse_probe(tmp_path):
    # Create a dummy probe.txt
    probe_content = "1:\n30878\n2647871\n2:\n2059652\n"
    probe_file = tmp_path / "probe.txt"
    probe_file.write_text(probe_content)
    
    output_file = tmp_path / "probe.parquet"
    
    parse_probe(str(probe_file), str(output_file))
    
    assert output_file.exists()
    df = pd.read_parquet(output_file)
    assert len(df) == 3
    assert df.iloc[0]['movie_id'] == 1
    assert df.iloc[0]['user_id'] == 30878
    assert df.iloc[2]['movie_id'] == 2
