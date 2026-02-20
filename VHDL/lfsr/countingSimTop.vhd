-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.
 
LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE STD.TEXTIO.ALL;

ENTITY countingSimTop IS
    GENERIC (
        n              : INTEGER := 10;  -- Bit-widths of memory pointer, counter, etc.
        d              : INTEGER := 10;  -- Number of confComp modules
        z              : INTEGER := 0;   -- Zero padding for RSA
        classNumber    : INTEGER := 10;  -- Number of classes for memory image
        logInNum       : INTEGER := 3;   -- MuxCell, ceilingLOG2(#popCounters)
        dimensionSize  : INTEGER := 1000 -- Actual dimension size for padding
    );
    PORT (
        clk                : IN  STD_LOGIC;
        rst                : IN  STD_LOGIC;
        run                : IN  STD_LOGIC;
        hv                 : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
        update_valid       : IN  STD_LOGIC;
        updated_truth      : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
        updated_prediction : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
        ground_truth       : IN  INTEGER;
        predicted_label    : IN  INTEGER;
        done               : OUT STD_LOGIC;
        pointer            : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0);
        dout               : OUT STD_LOGIC_VECTOR(classNumber*(n+logInNum)-1 DOWNTO 0);
        update_done        : OUT STD_LOGIC
    );
END ENTITY countingSimTop;

ARCHITECTURE behavioral OF countingSimTop IS

    COMPONENT countingSim IS
        GENERIC (
            n           : INTEGER := 10;
            d           : INTEGER := 10;
            z           : INTEGER := 0;
            classNumber : INTEGER := 10;
            logInNum    : INTEGER := 3
        );
        PORT (
            clk        : IN  STD_LOGIC;
            rst        : IN  STD_LOGIC;
            run        : IN  STD_LOGIC;
            done       : IN  STD_LOGIC;
            reg1Update : IN  STD_LOGIC;
            reg1rst    : IN  STD_LOGIC;
            reg2Update : IN  STD_LOGIC;
            reg2rst    : IN  STD_LOGIC;
            muxSel     : IN  STD_LOGIC_VECTOR(logInNum DOWNTO 0);
            hv         : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
            CHV        : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
            pointer    : IN  STD_LOGIC_VECTOR(n-1 DOWNTO 0);
            dout       : OUT STD_LOGIC_VECTOR(n+logInNum-1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT SeqAdderCtrl IS
        GENERIC (
            ceilingLogPop : INTEGER := 3;
            nPop          : INTEGER := 8
        );
        PORT (
            clk         : IN  STD_LOGIC;
            rst         : IN  STD_LOGIC;
            run         : IN  STD_LOGIC;
            reg1Update  : OUT STD_LOGIC;
            reg1rst     : OUT STD_LOGIC;
            reg2Update  : OUT STD_LOGIC;
            reg2rst     : OUT STD_LOGIC;
            muxSel      : OUT STD_LOGIC_VECTOR(ceilingLogPop DOWNTO 0)
        );
    END COMPONENT;
 
    COMPONENT countingSimCtrl IS
        GENERIC (
            n : INTEGER := 10
        );
        PORT (
            clk      : IN  STD_LOGIC;
            rst      : IN  STD_LOGIC;
            run      : IN  STD_LOGIC;
            runOut   : OUT STD_LOGIC;
            done     : OUT STD_LOGIC;
            pointer  : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0)
        );
    END COMPONENT;

    -- New memory signals
    TYPE CHV_memory IS ARRAY (classNumber-1 DOWNTO 0) OF STD_LOGIC_VECTOR((2**n)*d-1 DOWNTO 0);
    TYPE CHV_memory_tosim IS ARRAY (classNumber-1 DOWNTO 0) OF STD_LOGIC_VECTOR(d-1 DOWNTO 0);
    
    SIGNAL CHV_TO_OUT : CHV_memory_tosim;
    SIGNAL CHV : CHV_memory;
    
    FILE CHV_file : TEXT OPEN READ_MODE IS "CHV_img.mif"; -- Specify your file name

    -- Control Signals
    SIGNAL dones, runOut : STD_LOGIC;
    SIGNAL reg1Update, reg1rst, reg2Update, reg2rst : STD_LOGIC;
    SIGNAL muxSel : STD_LOGIC_VECTOR(logInNum DOWNTO 0);
    SIGNAL point : STD_LOGIC_VECTOR(n-1 DOWNTO 0);
    SIGNAL allZeros : STD_LOGIC_VECTOR((2**(n))*d - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL padded_updated_truth      : STD_LOGIC_VECTOR((2**(n))*d - 1 DOWNTO 0);
    SIGNAL padded_updated_prediction : STD_LOGIC_VECTOR((2**(n))*d - 1 DOWNTO 0);

BEGIN

    padded_updated_truth <= allZeros((2**(n))*d - 1 - dimensionSize DOWNTO 0) & updated_truth;
    padded_updated_prediction <= allZeros((2**(n))*d - 1 - dimensionSize DOWNTO 0) & updated_prediction;

    -- Reading Memory File and Learning Update
    PROCESS (clk)
        VARIABLE mif_line    : LINE;
        VARIABLE temp_bv     : BIT_VECTOR((2**n)*d-1 DOWNTO 0);
        VARIABLE initialized : BOOLEAN := FALSE;
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '1' THEN
                update_done <= '0';
                -- Initialize from file
                IF NOT initialized THEN
                    FOR i IN 0 TO classNumber-1 LOOP
                        IF NOT endfile(CHV_file) THEN
                            readline(CHV_file, mif_line);
                            read(mif_line, temp_bv);
                            CHV(i) <= TO_STDLOGICVECTOR(temp_bv);
                        ELSE
                            CHV(i) <= (others => '0');
                        END IF;
                    END LOOP;
                    initialized := TRUE;
                END IF;
            ELSIF update_valid = '1' THEN
                -- Update CHV vectors with learning results
                CHV(ground_truth) <= padded_updated_truth;
                CHV(predicted_label) <= padded_updated_prediction;
                update_done <= '1';
            ELSE
                update_done <= '0';
            END IF;
        END IF;
    END PROCESS; 

    -- Extracting Correct CHV Segment
    concatECC: FOR I IN classNumber-1 DOWNTO 0 GENERATE
        classesECC: FOR k IN d-1 DOWNTO 0 GENERATE
            CHV_TO_OUT(I)(k) <= CHV(I)(TO_INTEGER(UNSIGNED(point)) + (2**n) * k);
        END GENERATE classesECC;
    END GENERATE concatECC;

    -- Sequential Adder Controller
    AdderCtrl : SeqAdderCtrl
        GENERIC MAP (
            ceilingLogPop => logInNum,
            nPop          => d
        )
        PORT MAP (
            clk        => clk,
            rst        => rst,
            run        => dones,
            reg1Update => reg1Update,
            reg1rst    => reg1rst,
            reg2Update => reg2Update,
            reg2rst    => reg2rst,
            muxSel     => muxSel
        );	

    -- CountingSim Instances
    countSimArr: FOR I IN classNumber-1 DOWNTO 0 GENERATE
        comp : countingSim 
            GENERIC MAP (
                n         => n,
                d         => d,
                z         => z,
                classNumber => I,
                logInNum  => logInNum
            )
            PORT MAP (
                clk        => clk,
                rst        => rst,
                run        => runOut,
                done       => dones,
                reg1Update => reg1Update,
                reg1rst    => reg1rst,
                reg2Update => reg2Update,
                reg2rst    => reg2rst,
                muxSel     => muxSel,
                hv         => hv,
                CHV        => CHV_TO_OUT(I),
                pointer    => point,
                dout       => dout(((I+1)*(n+logInNum))- 1 DOWNTO (I*(n+logInNum)))
            );
    END GENERATE countSimArr;
		
    -- CountingSim Controller
    CompCtrl : countingSimCtrl
        GENERIC MAP (
            n => n
        )
        PORT MAP (
            clk     => clk,
            rst     => rst,
            run     => run,
            runOut  => runOut,
            done    => dones,
            pointer => point
        );

    -- Output Assignments
    pointer <= point;
    done    <= reg2Update;
	
END ARCHITECTURE behavioral;
