ps aux |grep scala |awk '{print $2}' |xargs kill
ps aux |grep java |awk '{print $2}' |xargs kill
ps aux |grep poly |awk '{print $2}' |xargs kill
ps aux | grep "sbt" | awk '{print $2}' | xargs kill
ps aux | grep naproche | awk '{print $2}' | xargs kill