from youtubesearchpython import VideosSearch
import webbrowser


def youtube_search(label):
    txt = "".join(label.split('  ---->  '))
    videos_search = VideosSearch(txt, limit=1)
    url = videos_search.result()['result'][0]['link']
    webbrowser.open_new_tab(url)

