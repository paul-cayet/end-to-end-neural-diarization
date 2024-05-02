input_dir="/mounts/Datasets3/2024-Diarization/INA_Snowden/medias_reencoded/tv"
output_dir="/folder/.../data"

for folder in "$input_dir"/*; do
    echo $folder
    for folder in "$folder"/*; do
        echo $folder
        for audio_file in "$folder"/*.wav; do
            filename=$(basename "$audio_file" .MPG.wav)
            # echo $filename
            # if [ -f "$audio_file" ]; then
            if [[ "$filename" == "130607FR20100_B" || "$filename" == "130612FR22100_B" ]]; then
                echo $filename
                ffmpeg -i "$audio_file" -f segment -segment_time 60 -c copy "$output_dir/${filename}_%03d.MPG.wav"
            fi
        done
    done
done