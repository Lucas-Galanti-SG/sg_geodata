import duckdb
con = duckdb.connect()
p = "data/processed/empresas.parquet"
print("Columns:", con.execute(f"DESCRIBE SELECT * FROM '{p}' LIMIT 1").fetchdf()["column_name"].tolist())
print("Count:", con.execute(f"SELECT count(*) FROM '{p}'").fetchone()[0])
print("Sample:")
print(con.execute(f"SELECT cnpj_basico, razao_social, porte, capital_social FROM '{p}' LIMIT 3").fetchdf())
