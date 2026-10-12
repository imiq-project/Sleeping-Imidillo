# Imidillo report, September 2026

I looked through Magdeburg’s sensor data for 1 to 30 September 2026 and compared it with August 2026. The clearest signals were repeated NO2 daily-limit exceedances at several air stations, a marked drop in ozone at **West**, and a few sharper shifts in EV charging use. I can say what moved and what did not, but the data do not tell me why.

## Highlights

- The WHO daily NO2 guideline of 25 µg/m³ was exceeded on 11 days at **City Tunnel**; the worst daily mean was 49.8 µg/m³.
- The same NO2 guideline was also exceeded on 11 days at **Schleinufer**; the worst daily mean was 35.0 µg/m³.
- At **Guericke Strasse**, the WHO daily NO2 guideline was exceeded on 8 days; the worst daily mean was 38.9 µg/m³.
- O3 at **West** was cleaner than in August by 15.3 µg/m³ at equal hour and weekday; the raw mean moved from 73.4 to 58.3 µg/m³.
- EV charging at **Leipziger Straße 45** was busier than in August by 16.7 percentage points at equal hour and weekday; the raw mean moved from 9.1% to 25.6%.

## Parking

I could use 4 of 9 parking sites this time. Across those four, a typical weekday was emptier overnight and became fuller through the morning, reaching 48.6% at 12:00 before easing back. The quietest point of the day was 03:00 at 23.8%.

Weekend behaviour was not uniform across sites. **Tiefgarage Hotel Maritim** was emptier on weekends by 13.5 percentage points, and **Tiefgarage Ulrichshaus** was emptier by 15.0 percentage points. **Elbauenpark - Messeparkplatz** and **Tiefgarage Friedensplatz** showed no measurable weekend change.

Compared with August, only **Tiefgarage Friedensplatz** was fuller, by 4.1 percentage points at equal hour and weekday; its raw mean moved from 26.6% to 31.3%. **Elbauenpark - Messeparkplatz**, **Tiefgarage Hotel Maritim**, and **Tiefgarage Ulrichshaus** showed no measurable change.

## EV charging

I had usable data from 17 of 18 charging stations. On a typical weekday, occupancy was lowest at 03:00 with 6.7%, climbed through the morning, and peaked at 11:00 with 39.8%. After that it eased through the afternoon and evening.

Weekend behaviour varied from station to station. Some stations were less busy on weekends, while many showed no measurable weekend change.

For the 15 stations I could compare with August, 4 were busier, 1 was less busy, and 10 showed no measurable change. The clearest rise was at **Leipziger Straße 45**, which was busier by 16.7 percentage points at equal hour and weekday; its raw mean moved from 9.1% to 25.6%. **Hegelstraße 4** was also busier by 13.9 percentage points, and it had no free point for 34.5% of known time. One station moved the other way: **Julius-Bremer-Straße 1** was less busy by 9.0 percentage points.

## Traffic

I could use 12 of 29 traffic streets. The typical weekday pattern was fastest at 04:00 with 52.7 km/h and slowest at 14:00 with 32.3 km/h. In plain terms, streets were usually faster very early in the morning and slower from late morning into the afternoon.

Weekend behaviour again differed by street. Several streets were faster on weekends, while others showed no measurable weekend change.

Compared with August, almost all usable streets showed no measurable change. The only clear month-on-month shift was at **Askanischer Platz**, which was slower by 1.7 km/h at equal hour and weekday; the raw mean moved from 39.2 km/h to 37.4 km/h. No usable street was faster than in August.

## Air quality

For NO2, all 4 stations were usable. Across them, a typical weekday was lowest at 04:00 with 12.8 µg/m³ and highest at 21:00 with 30.0 µg/m³. Weekend NO2 was lower at **Guericke Strasse** by 5.1 µg/m³ and at **Schleinufer** by 5.9 µg/m³, while **City Tunnel** and **West** showed no measurable weekend change. No NO2 station exceeded the EU hourly limit of 200 µg/m³. The WHO daily guideline of 25 µg/m³ was exceeded on 11 days at **City Tunnel**, 11 days at **Schleinufer**, 8 days at **Guericke Strasse**, and 1 day at **West**. Compared with August, no NO2 station showed a measurable change.

For O3, **West** was the only usable station. A typical weekday there was lowest at 08:00 with 36.4 µg/m³ and highest at 18:00 with 86.7 µg/m³. Weekend O3 showed no measurable change. The EU 8-hour target of 120 µg/m³ was not exceeded. The WHO 8-hour guideline of 100 µg/m³ was exceeded on 4 days, with a worst 8-hour mean of 106.6 µg/m³. Compared with August, **West** was cleaner by 15.3 µg/m³ at equal hour and weekday; the raw mean moved from 73.4 to 58.3 µg/m³.

I could not use any PM10 station: **GuerickeStrasse**, **Schleinufer**, and **West** were excluded because the value stayed unchanged from the previous hour in 55%, 52%, and 62% of hours.

I could not use any PM2.5 station for the same reason: **GuerickeStrasse**, **Schleinufer**, and **West** stayed unchanged from the previous hour in 64%, 63%, and 70% of hours.

## Data quality

For parking, I excluded **Allee-Center**, **City-Carre**, and **Domviertel** because coverage was below 60%, and **Hotel-Ratswaage** plus **Universitaetsplatz** because they had no data in this window. **Hotel-Ratswaage** and **Universitaetsplatz** were silent all month. Each usable parking site had a longest gap of 69 h. At **Tiefgarage Friedensplatz**, the observed maximum freeSpots was 114 although totalSpots was 90, so I used 114 as capacity.

For EV charging, **AnderBuckauerFaehre1** had no data and was silent all month. The longest EV gaps were 113 h at **Gareisstraße 12** and 95 h at **Berliner Chaussee 11**.

For traffic, 17 streets were excluded because coverage was below 60%. Usable traffic streets had gaps of 68 h at most sites, 69 h at **Hohepfortestrasse** and **Markgrafenstrasse**, and 70 h at **Schleinufer**. I also dropped implausible readings above 150 km/h at many traffic sites, including 36 at **Universitaetsplatz**, 32 at **Gareisstrasse**, and 26 at **Askanischer Platz**, **Markgrafenstrasse**, and **Wittenberger Strasse**. Zero readings were ignored 68 times at **Askanischer Platz**, 18 times at **Hohepfortestrasse**, and 11 times at **Erzbergerstrasse**.

For air stations, the longest gaps were 69 h at **City Tunnel**, **Guericke Strasse**, and **West**, and 71 h at **Schleinufer**. All PM10 and PM2.5 stations were excluded as stale.

Imidillo