
# Script to convert .mp4, .avi video to .gif format, which is then embedded to GitHub page
ffmpeg -i input_video.mp4 -t duration_time_to_convert output_video.gif
