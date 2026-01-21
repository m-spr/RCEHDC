LIBRARY ieee;
    USE ieee.std_logic_1164.ALL;
    USE ieee.numeric_std.ALL;

ENTITY binarizer IS
    PORT (
        start  : IN  std_logic;
        value  : IN  integer;
        result : OUT std_logic
    );
END ENTITY;

ARCHITECTURE behavioral OF binarizer IS
BEGIN
    PROCESS (start, value)
    BEGIN
        IF start = '1' THEN
            IF value >= 0 THEN
                result <= '1';
            ELSE
                result <= '0';
            END IF;
        END IF;
    END PROCESS;
END ARCHITECTURE;
