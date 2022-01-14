from preprocess import _for_all_files
from os import rename, path

BASE_DIR = "data/3s5s"

renameTable = [
    ("aim_chat_", "chat_aim_chat_"),
    ("aim_chat", "chat_aim_chat_"),
    ("email", "email_email_"),
    ("facebook_audio", "voip_facebook_"),
    ("facebook_chat_", "chat_facebook_"),
    ("facebook_chat", "chat_facebook_"),
    ("facebook_video", "video_facebook_"),
    ("ftps_down", "filetransfer_ftps_down"),
    ("ftps_up", "filetransfer_ftps_up"),
    ("hangout_chat", "chat_hangouts"),
    ("hangouts_audio", "voip_hangouts_"),
    ("hangouts_chat_", "chat_hangouts_"),
    ("hangouts_chat", "chat_hangouts_"),
    ("hangouts_video", "video_hangouts_"),
    ("icq_chat_", "chat_icq_chat_"),
    ("netflix_", "streaming_netflix_"),
    ("netflix", "streaming_netflix_"),
    ("scpDown", "filetransfer_scp_down_"),
    ("scpUp", "filetransfer_scp_up_"),
    ("sftp_down", "filetransfer_sftp_down"),
    ("sftp_up", "filetransfer_sftp_up"),
    ("sftpDown", "filetransfer_sftp_down_"),
    ("sftpUp", "filetransfer_sftp_up_"),
    ("skype_audio", "voip_skype_"),
    ("skype_chat", "chat_skype_"),
    ("skype_video", "video_skype_"),
    ("skype_files", "filetransfer_skype_"),
    ("spotify_", "streaming_spotify_"),
    ("spotify", "streaming_spotify_"),
    ("bittorrent", "p2p_bittorrent_"),
    ("ftps_", "filetransfer_ftps_"),
    ("sftp_", "filetransfer_sftp_"),
    ("icq_chat", "chat_icq_chat_"),
    ("vimeo_", "streaming_vimeo_"),
    ("vimeo", "streaming_vimeo_"),
    ("voipbuster_", "voip_voipbuster_"),
    ("voipbuster", "voip_voipbuster_"),
    ("youtube_", "streaming_youtube_"),
    ("youtube", "streaming_youtube_"),
]

if __name__ == "__main__":

    def process(filename, name, root, **kwargs):
        name = name.lower().replace("vpn_", "")
        for e in enumerate(renameTable):
            _, (pattern, replacement) = e
            pattern = pattern.lower()
            if name.startswith(pattern):
                name = name.replace(pattern, replacement)
                break
        name = name.replace("pcap_flow", "pcap_Flow")
        rename(filename, path.join(root, name))
        print(filename, " => ", path.join(root, name))

    _for_all_files(process, BASE_DIR)
