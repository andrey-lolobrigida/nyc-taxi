"""Baixa os dados brutos do projeto: parquets mensais do green taxi (NYC TLC)
e o shapefile de taxi zones.

Fonte: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Os arquivos são servidos pelo CloudFront da TLC.

Re-executável: arquivos já presentes em data/raw/ são pulados. Downloads são
atômicos (escreve em .part e renomeia no fim), então uma execução interrompida
nunca deixa um arquivo corrompido para trás.

Uso:
    python src/data_download.py [--start 2019-01] [--end 2021-12] [--dest data/raw]
"""

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
TRIP_DATA_URL = BASE_URL + "/trip-data/green_tripdata_{month}.parquet"
ZONES_SHAPEFILE_URL = BASE_URL + "/misc/taxi_zones.zip"
ZONES_LOOKUP_URL = BASE_URL + "/misc/taxi_zone_lookup.csv"

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "raw"

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5


def month_range(start: str, end: str) -> list[str]:
    """Lista de meses 'YYYY-MM' de start a end, inclusivos."""
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    if (start_year, start_month) > (end_year, end_month):
        raise ValueError(f"intervalo invertido: {start} > {end}")

    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def download(url: str, dest: Path) -> bool:
    """Baixa url para dest. Retorna True se baixou, False se já existia."""
    if dest.exists():
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url) as response, open(part, "wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            part.rename(dest)
            return True
        except (urllib.error.URLError, OSError) as exc:
            part.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  tentativa {attempt} falhou ({exc}); retry em {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01", help="primeiro mês (YYYY-MM)")
    parser.add_argument("--end", default="2021-12", help="último mês (YYYY-MM)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="diretório de saída")
    args = parser.parse_args()

    targets = [
        (TRIP_DATA_URL.format(month=month), args.dest / f"green_tripdata_{month}.parquet")
        for month in month_range(args.start, args.end)
    ]
    targets.append((ZONES_SHAPEFILE_URL, args.dest / "taxi_zones.zip"))
    targets.append((ZONES_LOOKUP_URL, args.dest / "taxi_zone_lookup.csv"))

    downloaded = skipped = 0
    for url, dest in targets:
        if download(url, dest):
            size_mb = dest.stat().st_size / 1024**2
            print(f"baixado  {dest.name}  ({size_mb:.1f} MB)")
            downloaded += 1
        else:
            print(f"já existe  {dest.name}")
            skipped += 1

    print(f"\n{downloaded} baixados, {skipped} já existiam, em {args.dest}")


if __name__ == "__main__":
    main()