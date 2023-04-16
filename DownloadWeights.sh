#/bin/bash 
# https://drive.google.com/file/d/1-B_98OxBS0U6ku4CNW4lB1L5GqickqhX/view?usp=share_link
echo "Downloading MobileNetv2 trained weights"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-B_98OxBS0U6ku4CNW4lB1L5GqickqhX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-B_98OxBS0U6ku4CNW4lB1L5GqickqhX" -O mbnet_epochs_50(1).h5 && rm -rf /tmp/cookies.txt
# https://drive.google.com/file/d/1107GfDt_Xgrlrxp6rd683LnSD5o5aYBC/view?usp=sharing
echo "Downloading vgg16 trained weights"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1107GfDt_Xgrlrxp6rd683LnSD5o5aYBC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1107GfDt_Xgrlrxp6rd683LnSD5o5aYBC" -O mbnet_epochs_50(1).h5 && rm -rf /tmp/cookies.txt
