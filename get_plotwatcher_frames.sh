## extract a frame from each folder within plotwatcher
## this is effectively a frame from each SD card

indir="/mnt/mbs-rec/gfa_plotwatch/plotwatcher_rawvids/"
outdir="/mnt/mbs-rec/gfa_plotwatch/plotwatcher_masks/"

echo "Looking for videos in $indir without frames/masks..."

for d in $indir/*/ ;
do
  ## check for existing frame file
  existframe="$outdir"framesNmasks/"$(basename $d)"_frame.png
  if [ ! -e "$existframe" ]
  then
    ## create new frame image
    echo "$(basename $d) doesnt exist"
    ## find any old image (that's over 450M so it's guaranteed to have a few mins of footage)
    vid=$(find $d -type f -size +500M -name "*.TLV" | head -1)
    ## grab one frame from the hh:mm:ss min mark
    echo "Grabbing frame from $vid..."
    ffmpeg -i $vid -ss 00:01:45 -vframes 1 "$outdir"framesNmasks/"$(basename $d)"_frame.png
  fi
done



