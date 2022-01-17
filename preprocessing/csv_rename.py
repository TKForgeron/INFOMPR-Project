from preprocessing.preprocessing import BASE_DIR_RENAMED, _for_all_files, BASE_DIR_RAW
from os import rename, path, makedirs

renameTable = [
    ("aim_chat_", "chat_aim_chat_"),
    ("aim_chat", "chat_aim_chat_"),
    ("email", "email_mail_"),
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
    ("skype_file", "filetransfer_skype_"),
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


def rename_all_in_folder():
    """Renames all csv files in the folder preprocess.BASE_DIR_RAW to the format used later by the tagging method."""

    def process(root, filename, name, **kwargs):
        originalName = name
        name = name.lower().replace("vpn_", "")
        for e in enumerate(renameTable):
            _, (pattern, replacement) = e
            pattern = pattern.lower()
            if name.startswith(pattern):
                name = name.replace(pattern, replacement)
                break
        name = name.replace("pcap_flow", "pcap_Flow")

        out_root = root.replace(BASE_DIR_RAW, BASE_DIR_RENAMED)
        if not path.exists(out_root):
            makedirs(out_root)
        rename(
            filename,
            filename.replace(originalName, name).replace(
                BASE_DIR_RAW, BASE_DIR_RENAMED
            ),
        )
        print(f"{filename} => {filename.replace(originalName, name)}")

    _for_all_files(process, BASE_DIR_RAW)


if __name__ == "__main__":
    rename_all_in_folder()
