import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent


def download_gh_folder(
    owner: str,
    repo: str,
    path: str,
    destination: str,
    branch: str = "main",
    exclude=None,
):
    exclude = [] if exclude is None else exclude

    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)

    url = f"https://api.github.com/repos/{owner}/{repo}/contents{path}?ref={branch}"
    response = requests.get(url)
    response.raise_for_status()

    contents = response.json()
    for item in contents:
        if item["type"] == "file" and item["name"] not in exclude:
            file_url = item["download_url"]
            file_name = item["name"]

            file_response = requests.get(file_url)
            file_response.raise_for_status()

            file_path = destination / file_name
            with open(file_path, "wb") as f:
                f.write(file_response.content)
            yield (file_path, item["path"])
        elif item["type"] == "dir" and item["name"] not in exclude:
            subfolder = destination / str(item["name"])
            subfolder.mkdir(exist_ok=True)
            yield from download_gh_folder(
                owner,
                repo,
                f"{path}/{item['name']}",
                subfolder,
                branch,
                exclude=exclude,
            )


def download_msmarco(destination: str):
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
    )
    response = requests.get(url, stream=True)

    # Iterate over the chunks of the response content
    with tempfile.TemporaryFile(mode="wb+") as temp_file:
        chunk_size = 1024 * 1024  # 1MB chunks
        for chunk in tqdm(
            response.iter_content(chunk_size),
            total=int(response.headers["Content-length"]) // chunk_size,
            desc="Downloading MSMARCO",
            unit="MB",
        ):
            # Write the chunk to the temporary file
            temp_file.write(chunk)

        temp_file.seek(0)
        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(destination_path)


def main():
    files = download_gh_folder(
        "bmeviauac01",
        "datadriven",
        "/docs/en/lecture-notes",
        ROOT_DIR / "data/datadriven",
        branch="master",
    )

    questions = []
    for file_path, gh_path in (
        t := tqdm(files, desc=f"Downloading files", unit="file")
    ):
        t.set_postfix_str(gh_path)
        with open(file_path, "r+") as f:
            try:
                contents = f.read()
                q_ix = contents.index("## Questions to test your knowledge")
            except:
                continue
            f.truncate(len(contents[:q_ix]))
            questions.append(
                contents[q_ix:].lstrip("## Questions to test your knowledge").strip()
            )
    print("Download complete: datadriven")

    with open(ROOT_DIR / "data/datadriven/questions.md", "w") as f:
        f.write("\n".join(questions))
    print("Questions saved!")

    files = download_gh_folder(
        "brandonstarxel",
        "chunking_evaluation",
        "/chunking_evaluation/evaluation_framework/general_evaluation_data",
        ROOT_DIR / "data/general_evaluation_data",
        exclude=["questions_db", "chatlogs.md"],
    )
    for file_path, gh_path in (
        t := tqdm(files, desc=f"Downloading files", unit="file")
    ):
        t.set_postfix_str(gh_path)
    print("Download complete: chunking evaluation dataset")

    download_msmarco(ROOT_DIR / "data")
    print("Download complete: MS MARCO")


if __name__ == "__main__":
    main()
