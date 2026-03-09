import duckdb, pathlib
con = duckdb.connect()

aux = pathlib.Path("data/raw/Auxiliares")
f = str(list(aux.glob("*CNAECSV*"))[0]).replace("\\", "/")
print("File:", f)

for enc in ("CP1252", "ISO_8859_1", "windows-1252"):
    try:
        sql = f"""SELECT * FROM read_csv('{f}',
            header=false, sep=';', encoding='{enc}', ignore_errors=true,
            columns={{'column0':'VARCHAR','column1':'VARCHAR'}}) LIMIT 3"""
        r = con.execute(sql).fetchdf()
        print(f"{enc} OK: {r.to_string()}")
    except Exception as e:
        print(f"{enc} FAIL: {str(e)[:200]}")
