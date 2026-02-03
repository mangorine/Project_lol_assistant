from youtube_transcript_api import YouTubeTranscriptApi

ytt_api = YouTubeTranscriptApi()
video = ytt_api.fetch("Jh-eYTB1Ij0")

full_text = " ".join([t.text for t in video])
        
full_text = full_text.replace("\n", " ")
full_text = full_text.replace("  ", " ")

print(full_text)